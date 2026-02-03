import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import duckdb
    from pipeline.stage_ingest import ingest_all
    from pipeline import ensure_connection, RAW_SCHEMA
    return RAW_SCHEMA, ensure_connection, ingest_all, mo


@app.cell
def _(mo):
    mo.md("""
    # Étape 1 – Ingestion & Normalisation

    - Charge les CSV (locaux ou data.gouv) `caracteristiques` et `usagers`.
    - Écrit les tables brutes dans DuckDB (`raw.caracteristiques`, `raw.usagers`).
    - Crée `raw.accidents_nc` avec les colonnes spatio-temporelles nettoyées.
    """)
    return


@app.cell
def _(ingest_all, mo):
    stats = ingest_all()
    mo.md(
        f"""
        ## Résultat ingestion
        - `raw.caracteristiques`: **{stats['caracteristiques']}** lignes
        - `raw.usagers`: **{stats['usagers']}** lignes
        - `raw.accidents_nc`: **{stats['accidents_nc']}** lignes (Nouvelle-Calédonie)
        - Sources utilisées: {stats['carac_sources']} fichiers caractéristiques / {stats['usagers_sources']} fichiers usagers
        """
    )
    stats
    return


@app.cell
def _(RAW_SCHEMA, ensure_connection):
    conn = ensure_connection()
    preview = conn.execute(f"SELECT * FROM {RAW_SCHEMA}.accidents_nc LIMIT 5").df()
    preview
    return


if __name__ == "__main__":
    app.run()
