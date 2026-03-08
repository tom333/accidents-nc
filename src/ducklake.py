"""Client DuckLake pour l'environnement production."""

import os
import urllib.parse

import duckdb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Schema names as constants (avoid circular imports with assets packages)
BRONZE_SCHEMA = "bronze"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA = "gold"


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

        # 1. Configuration S3 via CREATE SECRET
        self._configure_s3(conn)

        # 2. Attachement du catalogue DuckLake
        self._attach_ducklake(conn)

        # 3. Initialisation des schémas
        self._ensure_schemas(conn)

        return conn

    def _configure_s3(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Configure l'accès au stockage S3 (RustFS)."""
        s3_url = os.getenv("AWS_ENDPOINT_URL", "https://rustfs.tgu.ovh")
        s3_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        s3_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        if not s3_access_key or not s3_secret_key:
            raise ValueError("AWS_ACCESS_KEY_ID et AWS_SECRET_ACCESS_KEY sont requis.")

        parsed_url = urllib.parse.urlparse(s3_url)
        s3_endpoint = parsed_url.netloc or parsed_url.path
        use_ssl = str(s3_url.startswith("https")).lower()

        conn.execute(f"""
            CREATE SECRET IF NOT EXISTS ducklake_s3 (
                TYPE S3,
                KEY_ID '{s3_access_key}',
                SECRET '{s3_secret_key}',
                ENDPOINT '{s3_endpoint}',
                USE_SSL {use_ssl},
                URL_STYLE 'path'
            );
        """)

    def _attach_ducklake(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Attache le catalogue PostgreSQL et le stockage S3."""
        pg_url = os.getenv("DUCKLAKE_DATABASE_URL")
        s3_data_path = os.getenv("DUCKLAKE_DATA_PATH", "s3://votre-bucket/ducklake-data")
        pg_password = os.getenv("POSTGRES_PASSWORD") or os.getenv("PGPASSWORD")

        if not pg_url:
            raise ValueError("La variable d'environnement DUCKLAKE_DATABASE_URL est requise.")

        # Compatibilite: certains manifests fournissent un DSN prefixe par "postgres:".
        # Le prefixe est deja ajoute dans la syntaxe ATTACH (ducklake:postgres:<dsn/url>).
        if pg_url.startswith("postgres:"):
            pg_url = pg_url[len("postgres:") :]

        # Si le DSN est au format key=value (sans URL), ajouter le mot de passe injecte via Secret.
        if "://" not in pg_url and "password=" not in pg_url.lower() and pg_password:
            pg_url = f"{pg_url} password={pg_password}"

        # Evite de casser la requete SQL si un secret contient des quotes.
        pg_url_sql = pg_url.replace("'", "''")
        s3_data_path_sql = s3_data_path.replace("'", "''")

        conn.execute(f"""
            ATTACH 'ducklake:postgres:{pg_url_sql}' AS ducklake
            (DATA_PATH '{s3_data_path_sql}');
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
_client: DuckLakeClient | None = None


def get_client() -> DuckLakeClient:
    """Retourne le client DuckLake (singleton)."""
    global _client
    if _client is None:
        _client = DuckLakeClient()
    return _client


def reset_client():
    """Reset le client."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
