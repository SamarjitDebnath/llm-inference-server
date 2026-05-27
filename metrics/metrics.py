from __future__ import annotations

from collections import deque
from statistics import mean
from typing import Deque, Dict


class BatchMetrics:
    def __init__(self, max_samples: int = 1000):
        self.queue_latencies: Deque[float] = deque(maxlen=max_samples)
        self.batch_sizes: Deque[float] = deque(maxlen=max_samples)
        self.token_throughputs: Deque[float] = deque(maxlen=max_samples)

    def record_queue_latency(self, latency_seconds: float) -> None:
        if latency_seconds < 0:
            return
        self.queue_latencies.append(latency_seconds)

    def record_batch_size(self, batch_size: int) -> None:
        if batch_size <= 0:
            return
        self.batch_sizes.append(float(batch_size))

    def record_token_throughput(self, tokens: int, elapsed_seconds: float) -> None:
        if elapsed_seconds <= 0 or tokens < 0:
            return
        self.token_throughputs.append(tokens / elapsed_seconds)

    def _average(self, values: deque[float]) -> float | None:
        return mean(values) if values else None

    def snapshot(self) -> Dict[str, float | None]:
        average_queue_latency = self._average(self.queue_latencies)
        average_batch_size = self._average(self.batch_sizes)
        average_token_throughput = self._average(self.token_throughputs)
        return {
            "average_queue_latency_ms": average_queue_latency * 1000.0 if average_queue_latency is not None else None,
            "average_batch_size": average_batch_size,
            "average_token_throughput_per_sec": average_token_throughput,
            "queue_latency_samples": len(self.queue_latencies),
            "batch_size_samples": len(self.batch_sizes),
            "throughput_samples": len(self.token_throughputs),
        }


metrics = BatchMetrics()
