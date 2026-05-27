"""Pytest configuration and shared fixtures"""
import asyncio
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def project_root():
    """Return the project root directory"""
    return Path(__file__).parent.parent


@pytest.fixture
def test_prompt():
    """Return a test prompt"""
    return "What is machine learning?"


@pytest.fixture
def test_prompts():
    """Return multiple test prompts for batch testing"""
    return [
        "What is machine learning?",
        "Explain deep learning in simple terms",
        "How does a neural network work?",
        "What is a transformer model?",
        "Describe gradient descent",
    ]
