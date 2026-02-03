import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import duckdb
    from pipeline.stage_features import build_feature_store
    from pipeline import ensure_connection, FEATURES_SCHEMA
    return FEATURES_SCHEMA, build_feature_store, duckdb, ensure_connection, mo


@app.cell
def _(mo):
    mo.md(
        """
        # Étape 2 – Enrichissement & Features
        
        - Charge `raw.accidents_nc` et filtre les observations valides.
        - Génère les échantillons négatifs équilibrés spatialement/Temporellement.
        - Associe les features routières pré-calculées (`routes_with_features.pkl`).
        - Ajoute les features temporelles avancées puis stocke `features.full_dataset`.
        """
    )
    return


@app.cell
def _(build_feature_store, mo):
    stats = build_feature_store()
    mo.md(
        f"""
        ## Table `features.full_dataset`
        - Lignes totales: **{stats['rows']}**
        - Positives (accidents): **{stats['positives']}**
        - Négatives synthétiques: **{stats['negatives']}**
        """
    )
    stats
    return (stats,)


@app.cell
def _(FEATURES_SCHEMA, duckdb, ensure_connection):
    conn = ensure_connection()
    preview = conn.execute(f"SELECT latitude, longitude, target FROM {FEATURES_SCHEMA}.full_dataset LIMIT 5").df()
    preview
    return


if __name__ == "__main__":
    app.run()
