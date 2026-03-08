"""Ingestion des données brutes (bronze layer)."""

from __future__ import annotations

import glob
from collections.abc import Sequence

from src.ducklake import get_client

from .schema import BRONZE_SCHEMA

# URLs des CSV data.gouv.fr (2019-2024)
CARAC_REMOTE_URLS: Sequence[str] = (
    "https://www.data.gouv.fr/fr/datasets/r/e22ba475-45a3-46ac-a0f7-9ca9ed1e283a",  # 2019
    "https://www.data.gouv.fr/fr/datasets/r/07a88205-83c1-4123-a993-cba5331e8ae0",  # 2020
    "https://www.data.gouv.fr/fr/datasets/r/85cfdc0c-23e4-4674-9bcd-79a970d7269b",  # 2021
    "https://www.data.gouv.fr/fr/datasets/r/5fc299c0-4598-4c29-b74c-6a67b0cc27e7",  # 2022
    "https://www.data.gouv.fr/fr/datasets/r/104dbb32-704f-4e99-a71e-43563cb604f2",  # 2023
    "https://www.data.gouv.fr/api/1/datasets/r/83f0fb0e-e0ef-47fe-93dd-9aaee851674a",  # 2024
)

USAGERS_REMOTE_URLS: Sequence[str] = (
    "https://www.data.gouv.fr/fr/datasets/r/36b1b7b3-84b4-4901-9163-59ae8a9e3028",  # 2019
    "https://www.data.gouv.fr/fr/datasets/r/78c45763-d170-4d51-a881-e3147802d7ee",  # 2020
    "https://www.data.gouv.fr/fr/datasets/r/ba5a1956-7e82-41b7-a602-89d7dd484d7a",  # 2021
    "https://www.data.gouv.fr/fr/datasets/r/62c20524-d442-46f5-bfd8-982c59763ec8",  # 2022
    "https://www.data.gouv.fr/fr/datasets/r/68848e2a-28dd-4efc-9d5f-d512f7dbe66f",  # 2023
    "https://www.data.gouv.fr/api/1/datasets/r/f57b1f58-386d-4048-8f78-2ebe435df868",  # 2024
)


def _format_sources(sources: Sequence[str]) -> str:
    """Formate les sources CSV pour la requête DuckDB."""
    if not sources:
        raise ValueError("Aucune source CSV fournie")
    return "[" + ", ".join(f"'{path}'" for path in sources) + "]"


def _discover_files(pattern: str) -> list[str]:
    """Recherche les fichiers CSV locaux."""
    return sorted(glob.glob(pattern))


def ingest_caracteristiques() -> dict[str, int]:
    """Ingère les CSV caractéristiques (2019-2024) dans bronze.caracteristiques."""
    client = get_client()
    conn = client.conn

    # Chercher fichiers locaux d'abord, sinon URLs distantes
    local_files = _discover_files("data/caracteristiques-*.csv")
    sources = local_files or CARAC_REMOTE_URLS
    formatted = _format_sources(sources)

    conn.execute(f"""
        CREATE OR REPLACE TABLE {BRONZE_SCHEMA}.caracteristiques AS
        SELECT *
        FROM read_csv_auto(
            {formatted},
            union_by_name = TRUE
        )
    """)

    count = conn.execute(f"SELECT COUNT(*) FROM {BRONZE_SCHEMA}.caracteristiques").fetchone()[0]

    return {
        "table": "caracteristiques",
        "rows": count,
        "sources": len(sources),
        "type": "local" if local_files else "remote",
    }


def ingest_usagers() -> dict[str, int]:
    """Ingère les CSV usagers (2019-2024) dans bronze.usagers."""
    client = get_client()
    conn = client.conn

    # Chercher fichiers locaux d'abord, sinon URLs distantes
    local_files = _discover_files("data/usagers-*.csv")
    sources = local_files or USAGERS_REMOTE_URLS
    formatted = _format_sources(sources)

    conn.execute(f"""
        CREATE OR REPLACE TABLE {BRONZE_SCHEMA}.usagers AS
        SELECT *
        FROM read_csv_auto(
            {formatted},
            union_by_name = TRUE
        )
    """)

    count = conn.execute(f"SELECT COUNT(*) FROM {BRONZE_SCHEMA}.usagers").fetchone()[0]

    return {
        "table": "usagers",
        "rows": count,
        "sources": len(sources),
        "type": "local" if local_files else "remote",
    }


def create_accidents_nc() -> dict[str, int]:
    """Crée bronze.accidents_nc : accidents Nouvelle-Calédonie (dep=988) nettoyés."""
    client = get_client()
    conn = client.conn

    conn.execute(f"""
        CREATE OR REPLACE TABLE {BRONZE_SCHEMA}.accidents_nc AS
        WITH base AS (
            SELECT
                c.Num_Acc,
                -- Construire datetime en nettoyant hrmn (remplacer :: par :)
                regexp_replace(
                    concat(
                        lpad(cast(c.jour AS VARCHAR), 2, '0'), '/',
                        lpad(cast(c.mois AS VARCHAR), 2, '0'), '/',
                        cast(c.an AS VARCHAR), ' ',
                        substr(lpad(cast(c.hrmn AS VARCHAR), 4, '0'), 1, 2), ':',
                        substr(lpad(cast(c.hrmn AS VARCHAR), 4, '0'), 3, 2), ':00'
                    ),
                    '::', ':'
                ) AS datetime_str,
                cast(replace(trim(c.lat), ',', '.') AS DOUBLE) AS latitude,
                cast(replace(trim(c.long), ',', '.') AS DOUBLE) AS longitude,
                c.atm,
                c.dep AS dep_brut
            FROM {BRONZE_SCHEMA}.caracteristiques c
            LEFT JOIN {BRONZE_SCHEMA}.usagers u ON c.Num_Acc = u.Num_Acc
            WHERE c.dep = '988'
        ),
        parsed AS (
            SELECT
                Num_Acc,
                strptime(datetime_str, '%d/%m/%Y %H:%M:%S') AS event_time,
                latitude,
                longitude,
                atm
            FROM base
        )
        SELECT
            Num_Acc,
            event_time AS datetime,
            latitude,
            longitude,
            atm,
            hour(event_time) AS hour,
            dayofweek(event_time) AS dayofweek,
            month(event_time) AS month,
            1 AS target
        FROM parsed
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
    """)

    count = conn.execute(f"SELECT COUNT(*) FROM {BRONZE_SCHEMA}.accidents_nc").fetchone()[0]

    return {"table": "accidents_nc", "rows": count}


def ingest_all() -> dict[str, any]:
    """Ingère toutes les données bronze (caractéristiques + usagers + accidents_nc)."""
    carac_stats = ingest_caracteristiques()
    usagers_stats = ingest_usagers()
    accidents_stats = create_accidents_nc()

    return {
        "caracteristiques": carac_stats,
        "usagers": usagers_stats,
        "accidents_nc": accidents_stats,
        "total_rows": carac_stats["rows"] + usagers_stats["rows"] + accidents_stats["rows"],
    }
