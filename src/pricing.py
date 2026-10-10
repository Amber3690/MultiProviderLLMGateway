MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
}

def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
    input_cost = (prompt_tokens / 1_000_000.0) * rates["input"]
    output_cost = (completion_tokens / 1_000_000.0) * rates["output"]
    return round(input_cost + output_cost, 6)