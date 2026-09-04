import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

from .agent_core import AgentMode, BiomedAgent
from .ollama_retry import RetryingChatOllama

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

MODEL_PATH = "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
MODEL_PATH = "./models/Meta-Llama-3.1-70B-Instruct-Q4_K_S.gguf"
OLLAMA_OPTIONS: dict[str, Any] = {
    "num_ctx": 128000,
    "num_predict": 32768,
    "num_batch": 256,
}


def env_value(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


MODEL_PROVIDER = "ollama"
MODEL_NAME = env_value("MODEL_NAME")
if MODEL_NAME is None:
    raise RuntimeError("Model not defined. Set MODEL_NAME in multiagent/.env.")
MODEL_TEMPERATURE = float(env_value("MODEL_TEMPERATURE") or "0.0")
STREAMING_ENABLED = False
DISABLE_STREAMING = True


def ollama_base_url() -> str | None:
    return env_value("OLLAMA_BASE_URL", "OLLAMA_HOST")


def ollama_client_kwargs(base_url: str | None) -> dict[str, dict[str, str]]:
    api_key = env_value("OLLAMA_API_KEY", "Ollama_API_KEY")
    if not base_url:
        return {}
    if not api_key:
        return {}
    return {"headers": {"Authorization": f"Bearer {api_key}"}}


def configure_ollama_environment() -> None:
    api_key = env_value("OLLAMA_API_KEY", "Ollama_API_KEY")
    if api_key:
        os.environ["OLLAMA_API_KEY"] = api_key


def build_llm(model_name: str, temperature: float = 0.0) -> BaseChatModel:
    base_url = ollama_base_url()
    configure_ollama_environment()
    return RetryingChatOllama(
        model=model_name,
        temperature=temperature,
        base_url=base_url,
        reasoning=False,
        disable_streaming=DISABLE_STREAMING,
        client_kwargs=ollama_client_kwargs(base_url),
        **OLLAMA_OPTIONS,
    )


llm = build_llm(MODEL_NAME, MODEL_TEMPERATURE)

agent_multi_agent_standard = BiomedAgent(
    llm=llm,
    checkpoint_path="checkpoints_full_weak.db",
    mode=AgentMode.MULTI_AGENT_STANDARD,
)
graph_full_weak = agent_multi_agent_standard.create_graph()

# --- baselines -----------------------------------------------------------

agent_python_agent_retry_standard = BiomedAgent(
    llm=llm,
    checkpoint_path="checkpoints_pi_retries_weak.db",
    mode=AgentMode.PYTHON_AGENT_RETRY_STANDARD,
)
graph_pi_retries_weak = agent_python_agent_retry_standard.create_graph()

agent_python_agent_oneshot = BiomedAgent(
    llm=llm,
    checkpoint_path="checkpoints_pi_single.db",
    mode=AgentMode.PYTHON_AGENT_ONESHOT,
)
graph_pi_single = agent_python_agent_oneshot.create_graph()

agent_llm_only = BiomedAgent(
    llm=llm,
    checkpoint_path="checkpoints_llm_only.db",
    mode=AgentMode.LLM_ONLY,
)
graph_llm_only = agent_llm_only.create_graph()

agent_direct_tool_agent_strict = BiomedAgent(
    llm=llm,
    checkpoint_path="checkpoints_single_agent_strong.db",
    mode=AgentMode.DIRECT_TOOL_AGENT_STRICT,
)
graph_single_agent_strong = agent_direct_tool_agent_strict.create_graph()

agent_direct_tool_agent_standard = BiomedAgent(
    llm=llm,
    checkpoint_path="checkpoints_single_agent_weak.db",
    mode=AgentMode.DIRECT_TOOL_AGENT_STANDARD,
)
graph_single_agent_weak = agent_direct_tool_agent_standard.create_graph()

agent_python_agent_retry_strict = BiomedAgent(
    llm=llm,
    checkpoint_path="checkpoints_pi_retries_strong.db",
    mode=AgentMode.PYTHON_AGENT_RETRY_STRICT,
)
graph_pi_retries_strong = agent_python_agent_retry_strict.create_graph()

agent_multi_agent_strict = BiomedAgent(
    llm=llm,
    checkpoint_path="checkpoints_full_strong.db",
    mode=AgentMode.MULTI_AGENT_STRICT,
)
graph_full_strong = agent_multi_agent_strict.create_graph()
