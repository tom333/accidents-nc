"""Configuration centralisée (production)."""
import os
from dataclasses import dataclass


@dataclass
class DuckLakeConfig:
    """Configuration DuckLake (production)."""

    # PostgreSQL (catalog)
    postgres_url: str
    
    # S3 (storage)
    s3_endpoint: str
    s3_bucket: str
    s3_access_key: str
    s3_secret_key: str
    s3_use_ssl: bool
    
    @classmethod
    def from_env(cls) -> "DuckLakeConfig":
        """Charge config depuis variables d'environnement."""
        postgres_url = os.getenv("POSTGRES_URL")
        if not postgres_url:
            raise ValueError("POSTGRES_URL environment variable is required")

        s3_endpoint = os.getenv("S3_ENDPOINT", "https://rustfs.tgu.ovh")
        s3_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        s3_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        if not s3_access_key or not s3_secret_key:
            raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required")

        return cls(
            postgres_url=postgres_url,
            s3_endpoint=s3_endpoint,
            s3_bucket=os.getenv("S3_BUCKET", "accidents"),
            s3_access_key=s3_access_key,
            s3_secret_key=s3_secret_key,
            s3_use_ssl=s3_endpoint.startswith("https://"),
        )


# Instance globale
_config: DuckLakeConfig | None = None


def get_config() -> DuckLakeConfig:
    """Retourne la config (singleton)."""
    global _config
    if _config is None:
        _config = DuckLakeConfig.from_env()
    return _config
