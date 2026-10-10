from typing import Optional
from pydantic import BaseModel

class GatewayError(Exception):
    """Base exception for gateway failures."""
    pass

class AuthenticationError(GatewayError):
    pass

class RateLimitExceededError(GatewayError):
    pass

class GatewayProviderError(GatewayError):
    pass

class ChunkTelemetry(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_seconds: float = 0.0
    ttft_ms: float = 0.0
    tokens_per_second: float = 0.0
    total_cost_usd: float = 0.0

class StreamChunk(BaseModel):
    delta_text: str = ""
    provider: str
    model: str
    is_final: bool = False
    telemetry: Optional[ChunkTelemetry] = None