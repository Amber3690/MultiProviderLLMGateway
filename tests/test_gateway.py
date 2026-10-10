import pytest
import tiktoken
from src.gateway import UnifiedLLMGateway
from src.schemas import StreamChunk, ChunkTelemetry

@pytest.fixture
def gateway():
    return UnifiedLLMGateway()

def test_prompt_token_estimation(gateway):
    """Test 2: Token Count Validation using tiktoken within +-10% tolerance."""
    prompt = "This is a deterministic test prompt designed to verify token counting accuracy across language models."
    model = "gpt-4o-mini"
    
    enc = tiktoken.encoding_for_model(model)
    actual_tokens = len(enc.encode(prompt))
    estimated_tokens = gateway._estimate_prompt_tokens(prompt, model)
    
    # Assert within 10% tolerance
    assert abs(actual_tokens - estimated_tokens) <= max(1, actual_tokens * 0.1)

def test_telemetry_sanity_calculation():
    """Test 3: Telemetry Sanity checking cost, speed, and TTFT properties."""
    telemetry = ChunkTelemetry(
        prompt_tokens=50,
        completion_tokens=25,
        total_tokens=75,
        ttft_ms=180.5,
        tokens_per_second=42.1,
        total_cost_usd=0.000125
    )
    
    assert telemetry.total_cost_usd > 0
    assert telemetry.tokens_per_second > 0
    assert telemetry.ttft_ms > 0
    assert telemetry.total_tokens == telemetry.prompt_tokens + telemetry.completion_tokens

@pytest.mark.asyncio
async def test_stream_chunk_schema():
    """Test 1: Unified Stream Protocol chunk schema integrity."""
    chunk = StreamChunk(
        delta_text="Hello world",
        provider="openai",
        model="gpt-4o-mini",
        is_final=False
    )
    assert chunk.delta_text == "Hello world"
    assert chunk.provider == "openai"
    assert not chunk.is_final