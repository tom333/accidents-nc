import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import duckdb
    from pipeline.stage_datasets import build_datasets, FEATURE_COLUMNS
    from pipeline import ensure_connection, DATASET_SCHEMA
    return DATASET_SCHEMA, FEATURE_COLUMNS, build_datasets, duckdb, ensure_connection, mo


@app.cell
def _(mo):
    mo.md(
        """
        # Étape 3 – Jeux d'entraînement / test
        
        - Consomme `features.full_dataset`.
        - Encode `atm`, supprime les lignes incomplètes et applique le split train/test stratifié.
        - Stocke `datasets.train`, `datasets.test`, `datasets.feature_metadata` + sauvegarde `atm_encoder.pkl` & `features.pkl`.
        """
    )
    return


@app.cell
def _(build_datasets, mo):
    stats = build_datasets()
    mo.md(
        f"""
        ## Jeux prêts
        - Train: **{stats['train_rows']}** lignes
        - Test: **{stats['test_rows']}** lignes
        - Features: **{stats['n_features']}**
        """
    )
    stats
    return (stats,)


@app.cell
def _(DATASET_SCHEMA, FEATURE_COLUMNS, duckdb, ensure_connection, mo):
    conn = ensure_connection()
    sample = conn.execute(f"SELECT * FROM {DATASET_SCHEMA}.train LIMIT 5").df()
    mo.md(f"Colonnes utilisées: {', '.join(FEATURE_COLUMNS)}")
    sample
    return


if __name__ == "__main__":
    app.run()
