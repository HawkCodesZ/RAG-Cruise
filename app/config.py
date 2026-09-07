from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM gateway
    openai_api_key: str
    openai_base_url: str
    chat_model: str = "gpt-5-mini"
    embedding_model: str = "text-embedding-3-large"
    max_tokens: int = 1500  # gpt-5-mini spends variable tokens on hidden
                             # reasoning before visible output; too low
                             # silently truncates to an empty answer

    # Vector store
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_name: str = "incident_tickets"

    # Retrieval behavior
    top_k: int = 5
    similarity_threshold: float = 0.35

    # Source data
    tickets_xlsx_path: str = "./data/dummy_data_investigationAgent_1.xlsx"
    chunks_snapshot_path: str = "./data/chunks_snapshot.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()