import os


def get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value


class Settings:
    def __init__(self) -> None:
        self.dashscope_api_key = get_env("DASHSCOPE_API_KEY")
        self.dashscope_base_url = get_env(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com"
        )
        self.dashscope_chat_model = get_env("DASHSCOPE_CHAT_MODEL", "qwen-max")
        self.dashscope_embed_model = get_env(
            "DASHSCOPE_EMBED_MODEL", "text-embedding-v2"
        )
        self.max_chunk_tokens = int(get_env("MAX_CHUNK_TOKENS", "400") or "400")
        self.chunk_overlap_tokens = int(
            get_env("CHUNK_OVERLAP_TOKENS", "60") or "60"
        )
        self.max_parent_tokens = int(get_env("MAX_PARENT_TOKENS", "3000") or "3000")
        self.top_k_parents = int(get_env("TOP_K_PARENTS", "5") or "5")
        self.vector_weight = float(get_env("VECTOR_WEIGHT", "0.3") or "0.3")
        self.llm_weight = float(get_env("LLM_WEIGHT", "0.7") or "0.7")


settings = Settings()
