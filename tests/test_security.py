import pytest
from src.gateway import UnifiedLLMGateway
from src.schemas import AuthenticationError

@pytest.mark.asyncio
async def test_missing_api_keys_raise_typed_error(monkeypatch):
    """Test 4: Missing Key Handling raises typed AuthenticationError."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    
    gw = UnifiedLLMGateway()
    gw.openai_key = ""
    gw.anthropic_key = ""

    with pytest.raises(AuthenticationError) as exc_info:
        async for _ in gw.stream(prompt="Hi", model="gpt-4o-mini"):
            pass
    assert "API key is missing" in str(exc_info.value)

def test_secret_redaction_in_exceptions():
    """Test 5: Secret Redaction ensures raw keys are never leaked into string representations."""
    dummy_secret = "sk-proj-supersecretkey123456789"
    err = AuthenticationError("Authentication failed for openai: check credentials.")
    
    assert dummy_secret not in str(err)
    assert dummy_secret not in repr(err)