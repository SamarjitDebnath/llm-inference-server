import asyncio
import time
import torch
from typing import Optional
from transformers import DynamicCache
from settings.settings import model_settings


class InferenceRequest:
    def __init__(self, prompt, max_tokens, temperature):
        self.prompt = prompt
        self.max_tokens = max_tokens if max_tokens is not None else model_settings.max_length
        self.temperature = temperature if temperature is not None else model_settings.temperature

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        self.future = loop.create_future()
        # Streaming queue for token-wise output
        self.queue = asyncio.Queue()
        self.enqueue_time = time.monotonic()
        self.deadline: float | None = None
        self.queue_latency_ms: float | None = None
        # State for continuous generation
        self.input_ids: Optional[torch.Tensor] = None  # will be set when added to scheduler
        self.attention_mask: Optional[torch.Tensor] = None  # will be set when added to scheduler
        self.generated_tokens: list[int] = []
        self.finished = False
        self.past: Optional[DynamicCache] = None  # KV cache past_key_values


class ActiveRequest:
    def __init__(self, request, original_index):
        from settings.settings import model_settings
        self.request = request
        self.original_index = original_index
        self.max_tokens = request.max_tokens if request.max_tokens is not None else model_settings.max_length
        self.temperature = request.temperature if request.temperature is not None else model_settings.temperature
        # ActiveRequest mirrors InferenceRequest state for convenience
        self.generated_tokens = []
        self.finished = False
        self.past: Optional[DynamicCache] = None