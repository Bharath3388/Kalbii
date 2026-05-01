from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    KMI_PASSPHRASE: str = "change-me-please-very-secret"
    KMI_API_KEYS: str = "demo-key"

    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB: str = "kalbii"

    REDIS_URL: str = "redis://localhost:6379/0"

    DATA_DIR: str = "./data/images"

    CV_BACKEND: str = "opencv"        # opencv | autoencoder | clip
    NLP_BACKEND: str = "vader"        # vader | transformers

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    @property
    def api_keys(self) -> set[str]:
        return {k.strip() for k in self.KMI_API_KEYS.split(",") if k.strip()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
