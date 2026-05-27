"""Latency and performance benchmarks for the inference server"""
import asyncio
import time
import statistics
import pytest
from typing import List
from dataclasses import dataclass
from unittest.mock import Mock

from logger import setup_logger

logger = setup_logger(__name__, level="INFO")


@dataclass
class LatencyMetrics:
    """Container for latency metrics"""
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    stdev_ms: float
    p95_ms: float
    p99_ms: float


class TestLatencyBenchmarks:
    """Benchmark tests for inference latency"""

    def measure_latency(self, func, *args, **kwargs) -> float:
        """Measure execution time in milliseconds"""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return (end - start) * 1000  # Convert to ms

    async def measure_latency_async(self, coro) -> float:
        """Measure async execution time in milliseconds"""
        start = time.perf_counter()
        await coro
        end = time.perf_counter()
        return (end - start) * 1000  # Convert to ms

    def analyze_latencies(self, latencies: List[float]) -> LatencyMetrics:
        """Analyze a list of latency measurements"""
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)

        return LatencyMetrics(
            min_ms=min(latencies),
            max_ms=max(latencies),
            mean_ms=statistics.mean(latencies),
            median_ms=statistics.median(latencies),
            stdev_ms=statistics.stdev(latencies) if n > 1 else 0.0,
            p95_ms=latencies_sorted[p95_idx],
            p99_ms=latencies_sorted[p99_idx],
        )

    def print_latency_report(self, name: str, metrics: LatencyMetrics):
        """Log a formatted latency report"""
        logger.info("%s", "\n" + "=" * 60)
        logger.info("Latency Report: %s", name)
        logger.info("%s", "=" * 60)
        logger.info("  Min:     %.2f ms", metrics.min_ms)
        logger.info("  Max:     %.2f ms", metrics.max_ms)
        logger.info("  Mean:    %.2f ms", metrics.mean_ms)
        logger.info("  Median:  %.2f ms", metrics.median_ms)
        logger.info("  StdDev:  %.2f ms", metrics.stdev_ms)
        logger.info("  P95:     %.2f ms", metrics.p95_ms)
        logger.info("  P99:     %.2f ms", metrics.p99_ms)
        logger.info("%s", "=" * 60)

    def test_tokenizer_latency(self, test_prompt):
        """
        Benchmark tokenizer latency.
        
        Expected: < 20ms for typical prompts
        """
        try:
            from tokenizer.tokenizer_service import tokenizer_service
        except ImportError:
            pytest.skip("Tokenizer service not available")

        latencies = []
        num_runs = 50

        for _ in range(num_runs):
            latency = self.measure_latency(
                tokenizer_service.encode,
                test_prompt
            )
            latencies.append(latency)

        metrics = self.analyze_latencies(latencies)
        self.print_latency_report("Tokenizer Encoding", metrics)

        # Assertion: mean latency should be < 20ms
        assert metrics.mean_ms < 20.0, \
            f"Tokenizer latency too high: {metrics.mean_ms:.2f}ms (expected < 20ms)"

    def test_batch_throughput(self, test_prompts):
        """
        Benchmark batch processing throughput.
        
        Simulates processing multiple prompts in sequence.
        """
        try:
            from tokenizer.tokenizer_service import tokenizer_service
        except ImportError:
            pytest.skip("Tokenizer service not available")

        batch_latencies = []
        num_batches = 10

        for _ in range(num_batches):
            start = time.perf_counter()
            for prompt in test_prompts:
                tokenizer_service.encode(prompt)
            end = time.perf_counter()
            batch_latencies.append((end - start) * 1000)  # ms

        metrics = self.analyze_latencies(batch_latencies)
        batch_size = len(test_prompts)
        throughput = (batch_size / (metrics.mean_ms / 1000))  # prompts per second

        logger.info("%s", "\n" + "=" * 60)
        logger.info("Batch Throughput Report")
        logger.info("%s", "=" * 60)
        logger.info("  Batch Size:          %d prompts", batch_size)
        logger.info("  Mean Batch Latency:  %.2f ms", metrics.mean_ms)
        logger.info("  Throughput:          %.1f prompts/sec", throughput)
        logger.info("%s", "=" * 60)

        # Assertion: mean batch latency should be < 50ms
        assert metrics.mean_ms < 50.0, \
            f"Batch latency too high: {metrics.mean_ms:.2f}ms (expected < 50ms)"

    @pytest.mark.asyncio
    async def test_batch_scheduler_queue_latency(self):
        try:
            from scheduler.batch_scheduler import BatchScheduler
            from scheduler.request import InferenceRequest
            from scheduler.request_queue import batch_request_queue
            from metrics.metrics import metrics
        except ImportError:
            pytest.skip("Batch scheduler module not available")

        metrics.queue_latencies.clear()
        # Drain the queue before using it for this test.
        while not batch_request_queue.queue.empty():
            batch_request_queue.queue.get_nowait()

        first = InferenceRequest(prompt="latency prompt 1", max_tokens=1, temperature=0.5)
        first.enqueue_time = time.monotonic() - 0.08
        second = InferenceRequest(prompt="latency prompt 2", max_tokens=1, temperature=0.5)
        second.enqueue_time = time.monotonic() - 0.04

        await batch_request_queue.put(first)
        await batch_request_queue.put(second)

        scheduler = BatchScheduler(Mock(), Mock(), max_batch_size=2, queue_timeout=0.5)
        batch = await scheduler._collect_batch()

        assert len(batch) == 2
        assert len(metrics.queue_latencies) >= 2
        assert all(latency >= 0 for latency in metrics.queue_latencies)
        assert batch[0] is first
        assert batch[1] is second

    def test_repeated_inference_stability(self, test_prompt):
        """
        Test latency stability over repeated calls.
        
        Ensures that latency doesn't degrade with repeated usage.
        """
        try:
            from tokenizer.tokenizer_service import tokenizer_service
        except ImportError:
            pytest.skip("Tokenizer service not available")

        # Warm up the tokenizer to avoid measuring any one-time initialization costs.
        for _ in range(5):
            tokenizer_service.encode(test_prompt)

        latencies = []
        num_runs = 100

        for _ in range(num_runs):
            latency = self.measure_latency(
                tokenizer_service.encode,
                test_prompt
            )
            latencies.append(latency)

        # Split into early and late runs to check for degradation
        early = latencies[:25]
        late = latencies[-25:]

        early_metrics = self.analyze_latencies(early)
        late_metrics = self.analyze_latencies(late)

        logger.info("Latency Stability Test (100 runs)")
        logger.info("  Early (runs 1-25):  mean=%.2fms", early_metrics.mean_ms)
        logger.info("  Late (runs 76-100): mean=%.2fms", late_metrics.mean_ms)
        logger.info("  Degradation:        %.2fms", late_metrics.mean_ms - early_metrics.mean_ms)

        # Assertion: latency should not degrade by more than 20%, with a sane lower bound
        # to account for sub-millisecond scheduling jitter on modern systems.
        degradation = late_metrics.mean_ms - early_metrics.mean_ms
        max_degradation = max(early_metrics.mean_ms * 0.2, 0.5)
        epsilon = 0.001  # Small tolerance for floating point comparison
        assert degradation <= max_degradation + epsilon, \
            f"Latency degradation too high: {degradation:.2f}ms (expected <= {max_degradation:.2f}ms)"

    def test_concurrent_request_latency(self, test_prompts):
        """
        Benchmark latency under concurrent requests.
        
        Simulates multiple concurrent requests.
        """
        try:
            from tokenizer.tokenizer_service import tokenizer_service
        except ImportError:
            pytest.skip("Tokenizer service not available")

        async def concurrent_encode():
            tasks = [
                asyncio.create_task(asyncio.to_thread(
                    tokenizer_service.encode, prompt
                ))
                for prompt in test_prompts
            ]
            await asyncio.gather(*tasks)

        num_runs = 10
        concurrent_latencies = []

        for _ in range(num_runs):
            latency = asyncio.run(self.measure_latency_async(concurrent_encode()))
            concurrent_latencies.append(latency)

        metrics = self.analyze_latencies(concurrent_latencies)
        concurrent_throughput = (len(test_prompts) * num_runs) / (metrics.mean_ms / 1000)

        logger.info("%s", "\n" + "=" * 60)
        logger.info("Concurrent Request Latency Report")
        logger.info("%s", "=" * 60)
        logger.info("  Concurrent Requests: %d", len(test_prompts))
        logger.info("  Mean Latency:        %.2f ms", metrics.mean_ms)
        logger.info("  Throughput:          %.1f prompts/sec", concurrent_throughput)
        logger.info("%s", "=" * 60)


class TestLoadPatterns:
    """Test various load patterns and stress scenarios"""

    def test_memory_under_load(self, test_prompt):
        """
        Test memory usage under load.
        
        Runs multiple sequential tokenizations to check for memory leaks.
        """
        try:
            from tokenizer.tokenizer_service import tokenizer_service
            import psutil
            import os
        except ImportError:
            pytest.skip("Required modules not available")

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run many iterations
        for _ in range(500):
            tokenizer_service.encode(test_prompt)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        logger.info("Memory Test (500 iterations)")
        logger.info("  Initial Memory: %.2f MB", initial_memory)
        logger.info("  Final Memory:   %.2f MB", final_memory)
        logger.info("  Increase:       %.2f MB", memory_increase)

        # Assertion: memory increase should be < 100MB
        assert memory_increase < 100.0, \
            f"Memory increased too much: {memory_increase:.2f}MB (expected < 100MB)"
