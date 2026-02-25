"""Client DuckLake pour l'environnement production."""
import os
import urllib.parse
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
        """Connection lazy (créée à la première utilisation)."""
        if self._conn is None:
            self._conn = self._create_connection()
        return self._conn
    
    def _create_connection(self) -> duckdb.DuckDBPyConnection:
        """Crée la connexion DuckLake (production)."""
        conn = duckdb.connect()
        
        # 1. Configuration S3 via CREATE SECRET (déclenche l'autoload de httpfs/aws)
        self._configure_s3(conn)
        
        # 2. Attachement du catalogue DuckLake (déclenche l'autoload de ducklake et postgres)
        self._attach_ducklake(conn)
        
        # 3. Initialisation des schémas
        self._ensure_schemas(conn)
        
        return conn

    def _configure_s3(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Configure l'accès au stockage S3 (RustFS) via la méthode sécurisée CREATE SECRET."""
        s3_url = os.getenv("S3_ENDPOINT", "https://rustfs.tgu.ovh")
        s3_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        s3_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        if not s3_access_key or not s3_secret_key:
            raise ValueError("AWS_ACCESS_KEY_ID et AWS_SECRET_ACCESS_KEY sont requis.")

        # Parsing propre pour extraire uniquement le domaine (requis par DuckDB)
        parsed_url = urllib.parse.urlparse(s3_url)
        s3_endpoint = parsed_url.netloc or parsed_url.path
        use_ssl = str(s3_url.startswith("https")).lower()

        conn.execute(f"""
            CREATE SECRET ducklake_s3 (
                TYPE S3,
                KEY_ID '{s3_access_key}',
                SECRET '{s3_secret_key}',
                ENDPOINT '{s3_endpoint}',
                USE_SSL {use_ssl},
                URL_STYLE 'path'
            );
        """)

    def _attach_ducklake(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Attache le catalogue PostgreSQL et le stockage S3 via l'extension DuckLake."""
        pg_url = os.getenv("POSTGRES_URL")
        # DuckLake a besoin de savoir où stocker les fichiers Parquet sur S3
        s3_data_path = os.getenv("S3_BUCKET", "s3://votre-bucket/ducklake-data") 
        
        if not pg_url:
            raise ValueError("La variable d'environnement POSTGRES_URL est requise.")
            
        # Syntaxe DuckLake officielle : lie le catalogue PostgreSQL aux données sur S3
        conn.execute(f"""
            ATTACH 'ducklake:postgres:{pg_url}' AS ducklake 
            (DATA_PATH '{s3_data_path}');
        """)
        conn.execute("USE ducklake")

    def _ensure_schemas(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Crée les schemas Bronze/Silver/Gold si besoin."""
        for schema in (BRONZE_SCHEMA, SILVER_SCHEMA, GOLD_SCHEMA):
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    
    def __enter__(self):
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """Ferme proprement la connexion."""
        if self._conn:
            self._conn.close()
            self._conn = None


# --- Singleton Pattern ---
_client: Optional[DuckLakeClient] = None

def get_client() -> DuckLakeClient:
    """Retourne le client DuckLake (singleton)."""
    global _client
    if _client is None:
        _client = DuckLakeClient()
    return _client

def reset_client():
    """Reset le client (utile pour les tests)."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
