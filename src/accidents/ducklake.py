"""Client DuckLake pour l'environnement production."""
import os
from typing import Optional

import duckdb

from .bronze.schema import BRONZE_SCHEMA
from .gold.schema import GOLD_SCHEMA
from .silver.schema import SILVER_SCHEMA


class DuckLakeClient:
    """Client DuckLake (production uniquement)."""
    
    def __init__(self):
        self._conn = None
    
    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Connection lazy (crée à la première utilisation)."""
        if self._conn is None:
            self._conn = self._create_connection()
        return self._conn
    
    def _create_connection(self) -> duckdb.DuckDBPyConnection:
        """Crée la connexion DuckLake (production)."""
        conn = self._connect_prod()
        self._ensure_schemas(conn)
        return conn

    def _ensure_schemas(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Crée les schemas Bronze/Silver/Gold si besoin."""
        for schema in (BRONZE_SCHEMA, SILVER_SCHEMA, GOLD_SCHEMA):
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    
    def _connect_prod(self) -> duckdb.DuckDBPyConnection:
        """Connexion DuckLake (PostgreSQL catalog + S3)."""
        conn = duckdb.connect()
        
        # PostgreSQL catalog
        pg_url = os.getenv("POSTGRES_URL")
        if not pg_url:
            raise ValueError("POSTGRES_URL environment variable is required")
        
        conn.execute(f"ATTACH '{pg_url}' AS ducklake (TYPE postgres)")
        conn.execute("USE ducklake")
        
        # S3 storage (RustFS)
        s3_endpoint = os.getenv("S3_ENDPOINT", "https://rustfs.tgu.ovh")
        s3_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        s3_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        if not s3_access_key or not s3_secret_key:
            raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required")

        use_ssl = s3_endpoint.startswith("https://")

        conn.execute(f"""
            SET s3_endpoint = '{s3_endpoint}';
            SET s3_use_ssl = {'true' if use_ssl else 'false'};
            SET s3_access_key_id = '{s3_access_key}';
            SET s3_secret_access_key = '{s3_secret_key}';
        """)
        
        return conn
    
    def __enter__(self):
        return self.conn
    
    def __exit__(self, *args):
        if self._conn:
            self._conn.close()
    
    def close(self):
        """Ferme la connexion."""
        if self._conn:
            self._conn.close()
            self._conn = None


# Instance globale (singleton)
_client: Optional[DuckLakeClient] = None


def get_client() -> DuckLakeClient:
    """Retourne le client DuckLake (singleton)."""
    global _client
    if _client is None:
        _client = DuckLakeClient()
    return _client


def reset_client():
    """Reset le client (utile pour tests)."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
