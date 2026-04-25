from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_docs"

    # Ollama
    ollama_model: str = "gemma2:2b"

    # История диалога
    history_length: int = 20

    # Embeddings
    embedding_model: str = "intfloat/multilingual-e5-small"

    # Chunking
    chunk_max_chars: int = 800
    chunk_min_chars: int = 200

    # Retrieval
    top_k_vector: int = 10
    top_k_bm25: int = 10
    top_k_final: int = 5

    # Storage
    data_dir: str = "./data"

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
