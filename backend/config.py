import os
import json
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Union

class Settings(BaseSettings):
    # API Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS Origins - Can be a JSON array or a comma-separated string
    cors_origins: Union[str, List[str]] = ["*"]

    # Chroma Configuration
    chroma_persist_dir: str = "data/chroma_db"
    chroma_collection_name: str = "intellidocs_papers"

    # LLM Configuration
    # Supported: "local", "openai", "gemini", "ollama", "dummy"
    llm_provider: str = "local"
    llm_model: str = "data/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    # API Credentials & URLs
    openai_api_key: str = ""
    gemini_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    def get_cors_origins(self) -> List[str]:
        if isinstance(self.cors_origins, list):
            return self.cors_origins
        try:
            # Try parsing as JSON array
            return json.loads(self.cors_origins)
        except json.JSONDecodeError:
            # Fallback to comma-separated list
            return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    # Pydantic Settings Config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
