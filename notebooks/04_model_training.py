import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pipeline.stage_modeling import run_training
    return mo, run_training


@app.cell
def _(mo):
    mo.md(
        """
        # Étape 4 – Modélisation & suivi MLflow
        
        - Charge `datasets.train/test` depuis DuckDB.
        - Tune CatBoost / LightGBM / XGBoost (Optuna) puis entraîne LogisticRegression, TabNet, MLP embeddings.
        - Loggue tous les runs dans MLflow, sauvegarde `accident_model.pkl`, `atm_encoder.pkl`, `features.pkl`.
        """
    )
    return


@app.cell
def _(mo, run_training):
    summary = run_training()
    mo.md(
        f"""
        ## Modèle sélectionné
        - Best model: **{summary['best_model']}**
        - Top recall:
        {''.join([f"  - {row['Modèle']}: {row['Recall']}\n" for row in summary['leaders']])}
        """
    )
    summary
    return (summary,)


if __name__ == "__main__":
    app.run()
