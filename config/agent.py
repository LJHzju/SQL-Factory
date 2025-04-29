from typing import Dict
from typing_extensions import TypedDict


class LLMConfig(TypedDict):
    selection: Dict
    generation: Dict
    expansion: Dict


llm_config: LLMConfig = {
    "selection": {
        "model": "Qwen2.5-Coder-14B-Instruct",
        "api_key": "0",
        "base_url": "http://127.0.0.1:8000/v1",
        "temperature": 0.3,
        "top_p": 0.3,
        "max_tokens": 8192 * 2
    },
    "generation": {"model": "gpt-4o", "max_retries": 3},
    "expansion": {
        "model": "Qwen2.5-Coder-14B-Instruct",
        "api_key": "0",
        "base_url": "http://127.0.0.1:8000/v1",
        "temperature": 0.8,
        "top_p": 0.8,
        "max_tokens": 8192 * 2,
    },
}
