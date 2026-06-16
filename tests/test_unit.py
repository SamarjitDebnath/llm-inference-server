"""Unit tests for core modules"""
import asyncio
import torch
import pytest
from unittest.mock import AsyncMock, Mock, patch


class TestSchedulerRequestQueue:
    """Unit tests for the request queue"""

    def test_import_request_queue(self):
        """Test that request queue can be imported"""
        try:
            from scheduler.request_queue import request_queue
            assert request_queue is not None
        except ImportError:
            pytest.skip("Scheduler module not available")

    def test_import_inference_request(self):
        """Test that InferenceRequest can be imported"""
        try:
            from scheduler.request import InferenceRequest
            assert InferenceRequest is not None
        except ImportError:
            pytest.skip("Scheduler module not available")


class TestBatchScheduler:
    """Unit tests for batch scheduling and latency metrics"""

    @pytest.mark.asyncio
    async def test_batch_scheduler_processes_active_requests(self):
        try:
            from scheduler.batch_scheduler import BatchScheduler
            from scheduler.request import InferenceRequest
            from metrics.metrics import metrics
        except ImportError:
            pytest.skip("Batch scheduler module not available")

        metrics.queue_latencies.clear()
        metrics.batch_sizes.clear()
        metrics.token_throughputs.clear()

        mock_engine = Mock()
        mock_engine.generate_batch = AsyncMock(return_value=["first-output", "second-output"])

        mock_tokenizer = Mock()
        mock_tokenizer.tokenizer = Mock(return_value={
            "input_ids": torch.tensor([[1, 2], [1, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
        })

        scheduler = BatchScheduler(mock_engine, mock_tokenizer, max_batch_size=2, queue_timeout=0.01)
        requests = [
            InferenceRequest(prompt="first prompt", max_tokens=2, temperature=0.7),
            InferenceRequest(prompt="second prompt", max_tokens=2, temperature=0.9),
        ]

        await scheduler.process_batch(requests)

        assert requests[0].future.done()
        assert requests[0].future.result() == "first-output"
        assert requests[1].future.done()
        assert requests[1].future.result() == "second-output"
        assert metrics.batch_sizes[-1] == 2
        assert metrics.token_throughputs[-1] >= 0
        mock_engine.generate_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_batch_compaction_in_inference_engine(self):
        """Test that batch compaction updates active_requests and handles early completion."""
        try:
            from engine.generator import InferenceEngine
            from scheduler.request import InferenceRequest
            from unittest.mock import MagicMock
        except ImportError:
            pytest.skip("InferenceEngine or InferenceRequest not available")

        # Initialize engine
        engine = InferenceEngine()

        # Mock the model and its configuration
        mock_model = MagicMock()
        mock_model.config.eos_token_id = 50256  # GPT-2 EOS token
        engine._model = mock_model

        # Setup requests with unequal max_tokens
        req_0 = InferenceRequest(prompt="short", max_tokens=1, temperature=0.7)
        req_1 = InferenceRequest(prompt="longer", max_tokens=3, temperature=0.9)
        requests = [req_0, req_1]

        # Prepare input_ids and attention_mask tensors
        input_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1], [1, 1]], dtype=torch.long)

        class MockModelOutput:
            def __init__(self, logits, past_key_values):
                self.logits = logits
                self.past_key_values = past_key_values

        mock_pkv = MagicMock()

        # Mock the model calls to simulate compaction:
        # Step 1: 2 requests active. Request 0 generates EOS (50256), Request 1 generates 100.
        # Step 2: 1 request active (Request 1). Generates 101.
        # Step 3: 1 request active (Request 1). Generates 102 (finishes max_tokens=3).
        call_count = 0
        def model_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                logits = torch.zeros((2, 1, 50257))
                logits[0, 0, 50256] = 100.0  # Force EOS for req_0
                logits[1, 0, 100] = 100.0    # Token 100 for req_1
                return MockModelOutput(logits, mock_pkv)
            elif call_count == 2:
                logits = torch.zeros((1, 1, 50257))
                logits[0, 0, 101] = 100.0    # Token 101 for req_1
                return MockModelOutput(logits, mock_pkv)
            else:
                logits = torch.zeros((1, 1, 50257))
                logits[0, 0, 102] = 100.0    # Token 102 for req_1
                return MockModelOutput(logits, mock_pkv)

        mock_model.side_effect = model_side_effect

        outputs = await engine.generate_batch(input_ids, attention_mask, requests)

        from tokenizer.tokenizer_service import tokenizer_service
        assert len(outputs) == 2
        assert outputs[0] == tokenizer_service.decode([50256])
        assert outputs[1] == tokenizer_service.decode([100, 101, 102])
        assert call_count == 3
        mock_pkv.batch_select_indices.assert_called_once()



class TestTokenizerService:
    """Unit tests for tokenizer service"""

    def test_tokenizer_encode(self, test_prompt):
        """Test tokenizer encoding"""
        try:
            from tokenizer.tokenizer_service import tokenizer_service
        except ImportError:
            pytest.skip("Tokenizer service not available")

        # encode() with default return_tensors=False returns a list
        tokens = tokenizer_service.encode(test_prompt)
        assert tokens is not None
        # Can be list or dict depending on tokenizer state
        assert isinstance(tokens, (list, tuple, dict))
        if isinstance(tokens, dict):
            assert 'input_ids' in tokens or len(tokens) > 0
        else:
            assert len(tokens) > 0

    def test_tokenizer_decode(self):
        """Test tokenizer decoding"""
        try:
            from tokenizer.tokenizer_service import tokenizer_service
        except ImportError:
            pytest.skip("Tokenizer service not available")

        test_tokens = [101, 1045, 2001, 102]  # Sample tokens
        decoded = tokenizer_service.decode(test_tokens)
        assert decoded is not None
        assert isinstance(decoded, str)

    def test_kv_cache_hit_miss_rate(self, test_prompt):
        """Test KV cache hit/miss behavior during scheduler steps."""
        try:
            from scheduler.continuous_scheduler import ContinuousScheduler
            from scheduler.request import InferenceRequest
            from engine.model_loader import model_loader
            from engine.generator import engine
            from tokenizer.tokenizer_service import tokenizer_service
        except ImportError:
            pytest.skip("Required modules not available")

        model_loader.load()
        tokenizer_service.load()

        scheduler = ContinuousScheduler(engine, tokenizer_service, max_batch_size=1, timeout=0.01)
        request = InferenceRequest(prompt=test_prompt, max_tokens=10, temperature=1.0)

        encoded = tokenizer_service.encode(request.prompt, return_tensors=True)
        request.input_ids = encoded["input_ids"].to(engine.device)
        request.attention_mask = encoded["attention_mask"].to(engine.device)
        request.past = None
        scheduler.active_requests = [request]

        misses = 0
        hits = 0
        total_steps = 3

        non_eos_token = 0
        if engine.eos_token_id == non_eos_token:
            non_eos_token = 1

        for _ in range(total_steps):
            batch = scheduler._prepare_batch()
            assert batch is not None
            _, _, past_key_values = batch
            if past_key_values is None:
                misses += 1
            else:
                hits += 1

            logits, new_past = engine.forward_step(*batch)
            next_tokens = torch.tensor([[non_eos_token]], dtype=torch.long, device=engine.device)
            asyncio.run(scheduler._dispatch_tokens(next_tokens, new_past))

        assert request.past is not None, "Expected request KV cache to be populated after the first step"
        assert misses == 1, f"Expected exactly one cache miss, got {misses}"
        assert hits == total_steps - 1, f"Expected {total_steps - 1} cache hits, got {hits}"


class TestStreamManager:
    """Unit tests for the stream manager and decoding"""

    @pytest.mark.asyncio
    async def test_stream_response_handles_multi_byte_characters(self):
        """Test that stream manager decodes multi-byte tokens across boundaries correctly."""
        try:
            from streaming.stream_manager import stream_response
            from scheduler.request import InferenceRequest
            from tokenizer.tokenizer_service import tokenizer_service
        except ImportError:
            pytest.skip("Required modules not available")

        # Make sure tokenizer is loaded to decode properly
        tokenizer_service.load()

        # Emoji test
        text = "Hi 😊"
        tokens = tokenizer_service.encode(text)
        assert len(tokens) > 1

        req = InferenceRequest(prompt="test prompt", max_tokens=10, temperature=0.7)

        for token in tokens:
            req.queue.put_nowait(token)
        req.queue.put_nowait("[DONE]")

        yielded_slices = []
        async for chunk in stream_response(req):
            yielded_slices.append(chunk)

        reconstructed = "".join(yielded_slices)
        assert "Hi" in reconstructed
        assert "\ufffd" not in reconstructed


class TestAPIStructure:
    """Unit tests for API structure"""

    def test_import_routes(self):
        """Test that routes can be imported"""
        try:
            from api.routes import router
            assert router is not None
        except ImportError:
            pytest.skip("API routes not available")

    def test_import_server(self):
        """Test that server app can be imported"""
        try:
            from api.server import app
            assert app is not None
        except ImportError:
            pytest.skip("API server not available")


class TestSettings:
    """Unit tests for settings and configuration"""

    def test_import_model_settings(self):
        """Test that model settings can be imported"""
        try:
            from settings.settings import model_settings
            assert model_settings is not None
        except ImportError:
            pytest.skip("Settings not available")

    def test_import_logging_settings(self):
        """Test that logging settings can be imported"""
        try:
            from settings.settings import logging_settings
            assert logging_settings is not None
        except ImportError:
            pytest.skip("Settings not available")

    def test_settings_have_required_attributes(self):
        """Test that settings have expected attributes"""
        try:
            from settings.settings import model_settings
            # Check for common model settings
            assert hasattr(model_settings, 'top_k') or \
                   hasattr(model_settings, 'top_p') or \
                   hasattr(model_settings, 'model_name')
        except ImportError:
            pytest.skip("Settings not available")


class TestLogger:
    """Unit tests for logging"""

    def test_logger_setup(self):
        """Test that logger can be set up"""
        try:
            from logger import setup_logger
            logger = setup_logger(__name__)
            assert logger is not None
        except ImportError:
            pytest.skip("Logger not available")

    def test_logger_methods(self):
        """Test that logger has expected methods"""
        try:
            from logger import setup_logger
            logger = setup_logger(__name__)
            assert callable(getattr(logger, 'debug', None))
            assert callable(getattr(logger, 'info', None))
            assert callable(getattr(logger, 'warning', None))
            assert callable(getattr(logger, 'error', None))
        except ImportError:
            pytest.skip("Logger not available")
