import sys
from pathlib import Path

# Add project root directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import asyncio
import streamlit as st
from src.gateway import UnifiedLLMGateway
from src.schemas import CustomGatewayError

st.set_page_config(
    page_title="Multi-Provider LLM Gateway",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Multi-Provider LLM Gateway")
st.caption("Unified Asynchronous Streaming & Real-Time Micro-Cent Telemetry")

# Cache persistent gateway instance
@st.cache_resource
def get_gateway():
    return UnifiedLLMGateway()

gateway = get_gateway()

# Sidebar: Provider Selection & Real-Time Metrics
with st.sidebar:
    st.header("⚙️ Configuration")
    provider = st.selectbox(
        "Provider",
        options=["OpenAI", "Anthropic"],
        index=0
    )

    if provider == "OpenAI":
        model = st.selectbox(
            "Model",
            options=["gpt-4o-mini", "gpt-4o"],
            index=0
        )
    else:
        model = st.selectbox(
            "Model",
            options=["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"],
            index=0
        )

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

    st.markdown("---")
    st.header("📊 Execution Telemetry")
    ttft_metric = st.empty()
    speed_metric = st.empty()
    tokens_metric = st.empty()
    cost_metric = st.empty()

    ttft_metric.metric("Time to First Token (TTFT)", "0.0 ms")
    speed_metric.metric("Generation Speed", "0.0 tok/s")
    tokens_metric.metric("Token Breakdown", "In: 0 | Out: 0")
    cost_metric.metric("Execution Cost", "$0.000000")

# Main Prompt Interface
prompt = st.text_area(
    "Enter your prompt:",
    height=120,
    placeholder="Explain how asynchronous generators work in Python..."
)

if st.button("Generate Stream", type="primary"):
    if not prompt.strip():
        st.warning("Please provide a prompt before generating.")
    else:
        response_container = st.empty()

        async def run_stream():
            collected_text = ""
            try:
                async for chunk in gateway.stream(prompt=prompt, model=model, temperature=temperature):
                    if not chunk.is_final:
                        collected_text += chunk.delta_text
                        response_container.markdown(collected_text + "▌")
                    else:
                        response_container.markdown(collected_text)
                        if chunk.telemetry:
                            t = chunk.telemetry
                            ttft_metric.metric("Time to First Token (TTFT)", f"{t.ttft_ms} ms")
                            speed_metric.metric("Generation Speed", f"{t.tokens_per_second} tok/s")
                            tokens_metric.metric("Token Breakdown", f"In: {t.prompt_tokens} | Out: {t.completion_tokens}")
                            cost_metric.metric("Execution Cost", f"${t.total_cost_usd:.6f}")
            except CustomGatewayError as exc:
                st.error(f"Gateway Error: {str(exc)}")
            except Exception as exc:
                st.error(f"Unexpected Error: {type(exc).__name__}")

        asyncio.run(run_stream())