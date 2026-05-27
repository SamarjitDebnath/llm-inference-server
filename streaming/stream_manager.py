import asyncio
from typing import AsyncGenerator

from scheduler.request import InferenceRequest
from tokenizer.tokenizer_service import tokenizer_service
from settings.settings import logging_settings
from logger import setup_logger

logger = setup_logger(__name__, level=logging_settings.log_level, log_file=logging_settings.log_file)


def _decode_token(token_id: int) -> str:
    """Decode a single token ID to a string using the tokenizer service."""
    return tokenizer_service.decode([token_id])


async def stream_response(req: InferenceRequest) -> AsyncGenerator[str, None]:
    """Yield decoded tokens from an ``InferenceRequest``'s streaming queue.

    The generator reads token IDs from ``req.queue`` until the sentinel ``"[DONE]"``
    is received, decoding each token to a string before yielding it. This provides a
    clean, high‑level streaming API for the rest of the codebase (e.g., HTTP or
    WebSocket handlers).
    """
    while True:
        token = await req.queue.get()
        logger.debug("Stream manager received token=%s for prompt=%s", token, req.prompt)
        if token == "[DONE]":
            logger.debug("Stream manager received DONE for prompt=%s", req.prompt)
            break

        # Load tokenizer lazily and skip any special token emissions.
        if tokenizer_service.tokenizer is None:
            tokenizer_service.load()

        special_ids = getattr(tokenizer_service.tokenizer, "all_special_ids", None)
        if special_ids is not None and token in special_ids:
            logger.debug("Stream manager skipping special token=%s for prompt=%s", token, req.prompt)
            continue

        decoded = _decode_token(token)
        if decoded == "":
            logger.debug("Stream manager decoded empty string for token=%s prompt=%s", token, req.prompt)
            continue

        logger.debug(
            "Stream manager decoded token=%s -> '%s' for prompt=%s",
            token,
            decoded,
            req.prompt,
        )
        yield decoded
