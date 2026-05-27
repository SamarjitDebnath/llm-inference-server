import asyncio
import torch
from typing import Any, Tuple, cast
from transformers import DynamicCache
from settings.settings import model_settings, logging_settings
from tokenizer.tokenizer_service import tokenizer_service
from scheduler.request_queue import request_queue
from scheduler.request import InferenceRequest
from logger import setup_logger

logger = setup_logger(__name__, level=logging_settings.log_level, log_file=logging_settings.log_file)


class ContinuousScheduler:
    """Continuously processes inference requests with dynamic batching.

    This scheduler maintains a pool of active requests and processes them token-by-token.
    New requests can be added at any time (subject to ``max_batch_size``). It leverages
    the model's KV cache (``past_key_values``) to avoid recomputing the full sequence for
    each step, enabling O(n) inference per request. Tokens are streamed back to the caller
    via an ``asyncio.Queue`` attached to each :class:`InferenceRequest`.
    """

    def __init__(self, engine, tokenizer, max_batch_size: int = 8, timeout: float = 0.01):
        self.engine = engine
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        # List of active ``InferenceRequest`` objects
        self.active_requests: list[InferenceRequest] = []

    def _pad_batch(self, tensors, padding_value):
        max_len = max(t.size(1) for t in tensors)
        padded = []
        for t in tensors:
            if t.dim() != 2:
                raise RuntimeError(f"Unexpected tensor shape in _pad_batch: {tuple(t.shape)}")
            if t.size(1) < max_len:
                pad_amt = max_len - t.size(1)
                padded.append(torch.nn.functional.pad(t, (pad_amt, 0), value=padding_value))
            else:
                padded.append(t)
        return torch.cat(padded, dim=0)

    def _is_valid_dynamic_cache(self, past) -> bool:
        if past is None:
            return False
        try:
            for layer in past:
                if layer is None or len(layer) == 0:
                    return False
                if any(kv is None for kv in layer[:-1]):
                    return False
        except Exception:
            return False
        return True

    async def _add_new_requests(self):
        """Pull requests from the global ``request_queue`` until the batch is full
        or the timeout elapses.
        """
        while len(self.active_requests) < self.max_batch_size:
            try:
                req: InferenceRequest = await asyncio.wait_for(
                    request_queue.get(), timeout=self.timeout
                )
                # Tokenise the prompt once and move tensors to the engine device
                encoded = self.tokenizer.tokenizer(
                    req.prompt,
                    return_tensors="pt",
                )
                req.input_ids = encoded["input_ids"].to(self.engine.device)
                setattr(req, "attention_mask", encoded["attention_mask"].to(self.engine.device))
                setattr(req, "past", None)  # No KV cache for the first step
                self.active_requests.append(req)
                logger.debug(
                    "Added request to scheduler: prompt=%s, active_requests=%d",
                    req.prompt,
                    len(self.active_requests),
                )
            except asyncio.TimeoutError:
                break

    def _prepare_batch(self):
        """Build batched tensors from per-request state.

        Returns:
            ``(input_ids, attention_mask, past_key_values)`` ready for a
            forward pass, or ``None`` if no valid tensors exist.
        """
        for req in self.active_requests:
            if req.past is not None and not self._is_valid_dynamic_cache(req.past):
                logger.warning(
                    "Invalid stored DynamicCache detected for prompt=%s; falling back to full prompt batch.",
                    req.prompt,
                )
                req.past = None

        if any(r.past is None for r in self.active_requests):
            # First forward pass – use full prompts.
            input_tensors = [r.input_ids for r in self.active_requests if r.input_ids is not None]
            mask_tensors = [r.attention_mask for r in self.active_requests if r.attention_mask is not None]
            if not input_tensors or not mask_tensors:
                return None
            if len(input_tensors) > 1:
                logger.debug(
                    "Padding full prompt batch: shapes=%s",
                    [tuple(t.shape) for t in input_tensors],
                )
            input_ids = self._pad_batch(input_tensors, self.tokenizer.tokenizer.pad_token_id)
            attention_mask = self._pad_batch(mask_tensors, 0)
            past_key_values = None
        else:
            # Subsequent steps – only the most recent token per request.
            input_tensors = [r.input_ids[:, -1:] for r in self.active_requests if r.input_ids is not None]
            mask_tensors = [r.attention_mask[:, -1:] for r in self.active_requests if r.attention_mask is not None]
            if not input_tensors or not mask_tensors:
                return None
            input_ids = torch.cat(input_tensors, dim=0)
            attention_mask = torch.cat(mask_tensors, dim=0)
            # Stack each layer's KV cache across the batch dimension.
            # At this point we know no request has `past is None` (checked above).
            assert all(r.past is not None for r in self.active_requests)
            request_pasts = [cast(DynamicCache, r.past) for r in self.active_requests]
            if not request_pasts:
                return None

            batched_past_layers: list[tuple[Any, ...]] = []
            all_past_layers = [list(past) for past in request_pasts]
            num_layers = len(all_past_layers[0])

            for layer_idx in range(num_layers):
                layer_entries = [past_layers[layer_idx] for past_layers in all_past_layers]
                if any(layer_entry is None or len(layer_entry) == 0 for layer_entry in layer_entries):
                    return None

                batched_layer: list[Any] = []
                max_elems = max(len(layer_entry) for layer_entry in layer_entries)
                for elem_idx in range(max_elems):
                    elems = [layer_entry[elem_idx] if elem_idx < len(layer_entry) else None for layer_entry in layer_entries]
                    if any(elem is None for elem in elems):
                        batched_layer.append(None)
                    else:
                        if isinstance(elems[0], torch.Tensor):
                            batched_layer.append(torch.cat(cast(list[torch.Tensor], elems), dim=0))
                        else:
                            batched_layer.append(elems[0])
                batched_past_layers.append(tuple(batched_layer))

            past_key_values = DynamicCache(ddp_cache_data=batched_past_layers, config=self.engine.model.config)

        return input_ids, attention_mask, past_key_values

    async def _dispatch_tokens(self, next_tokens, new_past):
        """Stream sampled tokens back to clients and update per-request state.

        Args:
            next_tokens: Tensor of shape ``(batch, 1)`` with one token per
                active request.
            new_past: The full batched ``past_key_values`` returned by the
                model's forward pass.
        """
        logger.debug("Dispatching %d tokens to active requests", len(self.active_requests))
        finished_requests = []
        for idx, req in enumerate(self.active_requests):
            token_id = next_tokens[idx].item()

            # Stream token back to the client if the queue is configured
            if getattr(req, "queue", None) is not None:
                logger.debug("Sending token_id=%s to queue for prompt=%s", token_id, req.prompt)
                req.queue.put_nowait(token_id)

            # Record generated token
            req.generated_tokens.append(token_id)

            # Append token to tensors for the next iteration
            # Ensure tensors are present and use matching dtype/device for concatenation
            assert req.input_ids is not None
            new_token = torch.tensor([[token_id]], dtype=req.input_ids.dtype, device=req.input_ids.device)
            req.input_ids = torch.cat([req.input_ids, new_token], dim=1)

            attn = getattr(req, "attention_mask")
            assert attn is not None
            new_attn = torch.cat(
                [attn, torch.ones((1, 1), dtype=attn.dtype, device=attn.device)], dim=1
            )
            setattr(req, "attention_mask", new_attn)

            # Extract this request's slice of the new KV cache.
            # If any layer contains None, fall back to full prompt generation.
            if new_past is None:
                req_past = None
            else:
                def layer_cache_is_valid(layer_kv):
                    if layer_kv is None:
                        return False
                    if len(layer_kv) == 0:
                        return False
                    # Allow an optional trailing None placeholder, like
                    # (key_tuple, value_tuple, None).
                    if any(kv is None for kv in layer_kv[:-1]):
                        return False
                    return True

                past_contains_invalid = any(
                    not layer_cache_is_valid(layer_kv)
                    for layer_kv in new_past
                )
                if past_contains_invalid:
                    details = [
                        None if layer_kv is None else tuple(
                            None if kv is None else tuple(kv.shape for kv in kv)
                            for kv in layer_kv
                        )
                        for layer_kv in new_past
                    ]
                    logger.warning(
                        "Invalid past_key_values for prompt=%s; falling back to full prompt generation. structure=%s",
                        req.prompt,
                        details,
                    )
                    req_past = None
                else:
                    per_req_data: list[tuple[Any, ...]] = []
                    for layer_kv in new_past:
                        sliced_layer = []
                        for kv in layer_kv:
                            if kv is None:
                                sliced_layer.append(None)
                            elif isinstance(kv, torch.Tensor):
                                sliced_layer.append(kv[idx : idx + 1])
                            else:
                                sliced_layer.append(kv)
                        per_req_data.append(tuple(sliced_layer))

                    req_past = DynamicCache(ddp_cache_data=per_req_data, config=self.engine.model.config)

            setattr(req, "past", req_past)

            # Determine if the request has finished
            if (
                token_id == self.engine.eos_token_id
                or len(req.generated_tokens) >= req.max_tokens
            ):
                req.finished = True
                logger.info(
                    "Request finished for prompt=%s generated_length=%d finished=%s",
                    req.prompt,
                    len(req.generated_tokens),
                    req.finished,
                )
                if not req.future.done():
                    req.future.set_result(tokenizer_service.decode(req.generated_tokens))
                # Signal end of stream
                if getattr(req, "queue", None) is not None:
                    req.queue.put_nowait("[DONE]")
                finished_requests.append(req)

        # Remove finished requests from the active pool
        self.active_requests = [r for r in self.active_requests if r not in finished_requests]

    async def _step(self):
        """Run a single token generation step for *all* active requests.

        Orchestrates batch preparation, the engine's forward pass, penalty
        application, per-request sampling, and token dispatch.
        """
        # 1. Prepare batched tensors (scheduler concern)
        batch = self._prepare_batch()
        if batch is None:
            return
        input_ids, attention_mask, past_key_values = batch

        # 2. Forward pass (engine concern)
        logits, new_past = self.engine.forward_step(input_ids, attention_mask, past_key_values)

        # 3. Apply repetition penalty (engine concern)
        logits = self.engine.apply_repetition_penalty(logits, input_ids)

        # 4. Sample next token per request (engine concern)
        next_tokens = torch.stack([
            self.engine.sample(
                logits[i].unsqueeze(0),
                req.temperature,
                model_settings.top_k,
                model_settings.top_p,
            )
            for i, req in enumerate(self.active_requests)
        ])

        # 5. Dispatch tokens and update request state (scheduler concern)
        await self._dispatch_tokens(next_tokens, new_past)

    async def run(self):
        """Main scheduler loop.

        Repeatedly adds new requests, executes a token step for all active requests,
        and sleeps briefly when idle.
        """
        while True:
            try:
                await self._add_new_requests()
                if not self.active_requests:
                    await asyncio.sleep(0.01)
                    continue
                await self._step()
            except Exception as exc:
                logger.exception("Scheduler error during run loop: %s", exc)
                await asyncio.sleep(1)