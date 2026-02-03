from __future__ import annotations

from pathlib import Path
import glob
from typing import Iterable, Sequence

import duckdb

from .config import PIPELINE_PARAMS, BRONZE_SCHEMA, ensure_connection

CARAC_REMOTE_URLS: Sequence[str] = (
    "https://www.data.gouv.fr/fr/datasets/r/e22ba475-45a3-46ac-a0f7-9ca9ed1e283a",
    "https://www.data.gouv.fr/fr/datasets/r/07a88205-83c1-4123-a993-cba5331e8ae0",
    "https://www.data.gouv.fr/fr/datasets/r/85cfdc0c-23e4-4674-9bcd-79a970d7269b",
    "https://www.data.gouv.fr/fr/datasets/r/5fc299c0-4598-4c29-b74c-6a67b0cc27e7",
    "https://www.data.gouv.fr/fr/datasets/r/104dbb32-704f-4e99-a71e-43563cb604f2",
    "https://www.data.gouv.fr/api/1/datasets/r/83f0fb0e-e0ef-47fe-93dd-9aaee851674a",
)

USAGERS_REMOTE_URLS: Sequence[str] = (
    "https://www.data.gouv.fr/fr/datasets/r/36b1b7b3-84b4-4901-9163-59ae8a9e3028",
    "https://www.data.gouv.fr/fr/datasets/r/78c45763-d170-4d51-a881-e3147802d7ee",
    "https://www.data.gouv.fr/fr/datasets/r/ba5a1956-7e82-41b7-a602-89d7dd484d7a",
    "https://www.data.gouv.fr/fr/datasets/r/62c20524-d442-46f5-bfd8-982c59763ec8",
    "https://www.data.gouv.fr/fr/datasets/r/68848e2a-28dd-4efc-9d5f-d512f7dbe66f",
    "https://www.data.gouv.fr/api/1/datasets/r/f57b1f58-386d-4048-8f78-2ebe435df868",
)


def _format_sources(sources: Iterable[str]) -> str:
    paths = list(sources)
    if not paths:
        raise ValueError("Aucune source CSV fournie")
    return "[" + ", ".join(f"'{path}'" for path in paths) + "]"


def _discover_files(pattern: str) -> list[str]:
    return sorted(glob.glob(pattern))


def _ingest_table(conn: duckdb.DuckDBPyConnection, table: str, sources: Iterable[str]) -> int:
    formatted = _format_sources(sources)
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE {BRONZE_SCHEMA}.{table} AS
        SELECT *
        FROM read_csv_auto(
            {formatted},
            union_by_name = TRUE
        )
        """
    )
    return conn.execute(f"SELECT COUNT(*) FROM {BRONZE_SCHEMA}.{table}").fetchone()[0]


def ingest_all() -> dict[str, int]:
    conn = ensure_connection()
    carac_files = _discover_files('data/caracteristiques-*.csv')
    usagers_files = _discover_files('data/usagers-*.csv')

    carac_sources = carac_files or CARAC_REMOTE_URLS
    usagers_sources = usagers_files or USAGERS_REMOTE_URLS

    nb_carac = _ingest_table(conn, 'caracteristiques', carac_sources)
    nb_usagers = _ingest_table(conn, 'usagers', usagers_sources)

    conn.execute(
        f"""
        CREATE OR REPLACE TABLE {BRONZE_SCHEMA}.accidents_nc AS
        WITH base AS (
            SELECT
                c.Num_Acc,
                strptime(
                    concat(
                        lpad(cast(c.jour AS VARCHAR), 2, '0'), '/',
                        lpad(cast(c.mois AS VARCHAR), 2, '0'), '/',
                        cast(c.an AS VARCHAR), ' ',
                        substr(lpad(cast(c.hrmn AS VARCHAR), 4, '0'), 1, 2), ':',
                        substr(lpad(cast(c.hrmn AS VARCHAR), 4, '0'), 3, 2), ':00'
                    ),
                    '%d/%m/%Y %H:%M:%S'
                ) AS event_time,
                cast(replace(trim(c.lat), ',', '.') AS DOUBLE) AS latitude,
                cast(replace(trim(c.long), ',', '.') AS DOUBLE) AS longitude,
                c.atm,
                c.dep AS dep_brut
            FROM {BRONZE_SCHEMA}.caracteristiques c
            LEFT JOIN {BRONZE_SCHEMA}.usagers u ON c.Num_Acc = u.Num_Acc
            WHERE c.dep = '988'
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
        FROM base
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        """
    )
    nb_clean = conn.execute(f"SELECT COUNT(*) FROM {BRONZE_SCHEMA}.accidents_nc").fetchone()[0]

    return {
        'caracteristiques': nb_carac,
        'usagers': nb_usagers,
        'accidents_nc': nb_clean,
        'carac_sources': len(carac_sources),
        'usagers_sources': len(usagers_sources),
    }
