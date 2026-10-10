# AI Prompt Audit & Verification Log

## 1. Initial Prompt
```text
You are an expert Python AI Infrastructure Engineer. I need an asynchronous, streaming multi-provider SDK wrapper supporting both OpenAI (e.g., gpt-4o-mini, gpt-4o) and Anthropic (e.g., claude-3-5-sonnet, claude-3-5-haiku).

Requirements:
1. Environment: Python 3.11+, typed with Pydantic v2.
2. Architecture:
   - A unified class UnifiedLLMGateway with an async generator method:
     async def stream(self, prompt: str, model: str, temperature: float = 0.7) -> AsyncGenerator[StreamChunk, None]:
   - Use official async clients (AsyncOpenAI and AsyncAnthropic).
3. Normalization:
   - Define a Pydantic schema StreamChunk containing delta_text, provider, model, is_final, and telemetry.
4. Token & Cost Telemetry:
   - Capture TTFT, tokens/sec, prompt/completion tokens, and USD cost rounded to 6 decimal places.
5. Error Handling:
   - Raise typed exceptions (AuthenticationError, RateLimitExceededError, GatewayProviderError) without leaking raw keys.