import os
import time
from typing import AsyncGenerator
import tiktoken
from dotenv import load_dotenv

from openai import (
    AsyncOpenAI,
    AuthenticationError as OpenAIAuthError,
    RateLimitError as OpenAIRateLimit,
)
from anthropic import (
    AsyncAnthropic,
    AuthenticationError as AnthropicAuthError,
    RateLimitError as AnthropicRateLimit,
)

from src.schemas import (
    StreamChunk,
    ChunkTelemetry,
    AuthenticationError,
    RateLimitExceededError,
    GatewayProviderError,
)
from src.pricing import calculate_cost

load_dotenv()

class UnifiedLLMGateway:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

        # Persistent pooled client instances
        self.openai_client = AsyncOpenAI(api_key=self.openai_key or "missing")
        self.anthropic_client = AsyncAnthropic(api_key=self.anthropic_key or "missing")

    def _estimate_prompt_tokens(self, prompt: str, model: str) -> int:
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(prompt))

    async def stream(
        self, prompt: str, model: str, temperature: float = 0.7
    ) -> AsyncGenerator[StreamChunk, None]:
        is_openai = "gpt" in model.lower()
        provider = "openai" if is_openai else "anthropic"

        if is_openai and (not self.openai_key or self.openai_key == "missing"):
            raise AuthenticationError("OpenAI API key is missing or invalid.")
        if not is_openai and (not self.anthropic_key or self.anthropic_key == "missing"):
            raise AuthenticationError("Anthropic API key is missing or invalid.")

        start_time = time.perf_counter()
        first_token_time: float | None = None
        output_tokens = 0
        prompt_tokens = self._estimate_prompt_tokens(prompt, model)

        try:
            if is_openai:
                stream_resp = await self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                async for chunk in stream_resp:
                    if chunk.usage:
                        prompt_tokens = chunk.usage.prompt_tokens
                        output_tokens = chunk.usage.completion_tokens

                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta.content or ""
                        if delta:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            output_tokens += 1
                            yield StreamChunk(
                                delta_text=delta,
                                provider=provider,
                                model=model,
                                is_final=False,
                            )
            else:
                async with self.anthropic_client.messages.stream(
                    model=model,
                    max_tokens=2048,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                ) as stream_resp:
                    async for event in stream_resp:
                        if event.type == "message_start":
                            prompt_tokens = event.message.usage.input_tokens
                        elif event.type == "content_block_delta":
                            delta = event.delta.text
                            if delta:
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                output_tokens += 1
                                yield StreamChunk(
                                    delta_text=delta,
                                    provider=provider,
                                    model=model,
                                    is_final=False,
                                )
                        elif event.type == "message_delta":
                            if hasattr(event, "usage") and event.usage:
                                output_tokens = event.usage.output_tokens

        except (OpenAIAuthError, AnthropicAuthError):
            raise AuthenticationError(f"Authentication failed for {provider}: check credentials.") from None
        except (OpenAIRateLimit, AnthropicRateLimit):
            raise RateLimitExceededError(f"Rate limit exceeded for {provider}.") from None
        except Exception as exc:
            raise GatewayProviderError(f"Provider error encountered: {type(exc).__name__}") from None

        total_time = max(time.perf_counter() - start_time, 0.001)
        ttft_ms = round(((first_token_time - start_time) * 1000) if first_token_time else 0.0, 2)
        tokens_per_sec = round(output_tokens / total_time, 2)
        cost = calculate_cost(model, prompt_tokens, output_tokens)

        telemetry = ChunkTelemetry(
            prompt_tokens=prompt_tokens,
            completion_tokens=output_tokens,
            total_tokens=prompt_tokens + output_tokens,
            ttft_ms=ttft_ms,
            tokens_per_second=tokens_per_sec,
            total_cost_usd=cost,
        )

        yield StreamChunk(
            delta_text="",
            provider=provider,
            model=model,
            is_final=True,
            telemetry=telemetry,
        )