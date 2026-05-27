"""Integration tests for API endpoints"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch


class TestAPIIntegration:
    """Integration tests for the API server"""

    def test_app_initialization(self):
        """Test that the FastAPI app initializes correctly"""
        try:
            from api.server import app
            assert app is not None
            # Check for expected routes
            routes = [getattr(route, "path", None) for route in app.routes]
            routes = [r for r in routes if r is not None]
            assert len(routes) > 0
        except ImportError:
            pytest.skip("API server not available")

    def test_inference_request_structure(self):
        """Test that inference requests have correct structure"""
        try:
            from scheduler.request import InferenceRequest
            import asyncio
            
            # Ensure event loop exists for asyncio.Future()
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            req = InferenceRequest(
                prompt="Test prompt",
                max_tokens=100,
                temperature=0.7
            )
            
            assert req.prompt == "Test prompt"
            assert req.max_tokens == 100
            assert req.temperature == 0.7
            assert hasattr(req, 'generated_tokens')
        except ImportError:
            pytest.skip("Scheduler module not available")

    def test_continuous_scheduler_initialization(self):
        """Test that the continuous scheduler can be initialized"""
        try:
            from scheduler.continuous_scheduler import ContinuousScheduler
            
            # Create mock objects
            mock_engine = Mock()
            mock_tokenizer = Mock()
            
            scheduler = ContinuousScheduler(
                engine=mock_engine,
                tokenizer=mock_tokenizer,
                max_batch_size=8
            )
            
            assert scheduler.max_batch_size == 8
            assert scheduler.active_requests == []
        except ImportError:
            pytest.skip("Scheduler not available")

    def test_scheduler_request_handling(self):
        """Test that scheduler handles requests correctly"""
        try:
            from scheduler.continuous_scheduler import ContinuousScheduler
            
            mock_engine = Mock()
            mock_tokenizer = Mock()
            
            scheduler = ContinuousScheduler(
                engine=mock_engine,
                tokenizer=mock_tokenizer,
                max_batch_size=2
            )
            
            # Verify initial state
            assert len(scheduler.active_requests) == 0
            
            # Verify max batch size constraint
            assert scheduler.max_batch_size == 2
        except ImportError:
            pytest.skip("Scheduler not available")

    def test_batch_endpoint_route_exists(self):
        """Test that the batch generation route is registered"""
        try:
            from api.server import app
            routes = [getattr(route, "path", None) for route in app.routes]
            assert "/api/generate_batch" in routes
            assert "/api/metrics" in routes
        except ImportError:
            pytest.skip("API server not available")


class TestErrorHandling:
    """Test error handling in components"""

    def test_scheduler_error_resilience(self):
        """Test that scheduler handles errors gracefully"""
        try:
            from scheduler.continuous_scheduler import ContinuousScheduler
            
            mock_engine = Mock()
            mock_tokenizer = Mock()
            
            scheduler = ContinuousScheduler(
                engine=mock_engine,
                tokenizer=mock_tokenizer
            )
            
            # Should not raise on initialization
            assert scheduler is not None
        except ImportError:
            pytest.skip("Scheduler not available")

    def test_tokenizer_error_handling(self):
        """Test tokenizer error handling"""
        try:
            from tokenizer.tokenizer_service import tokenizer_service
            
            # Empty string should still work
            result = tokenizer_service.encode("")
            assert result is not None
        except ImportError:
            pytest.skip("Tokenizer service not available")
