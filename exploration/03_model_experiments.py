import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
    from src.accidents.ducklake import get_client
    return (
        RandomForestClassifier,
        LogisticRegression,
        classification_report,
        confusion_matrix,
        get_client,
        joblib,
        mo,
        np,
        pl,
        roc_auc_score,
    )


@app.cell
def _(mo):
    mo.md("""
    # Expérimentations Modèles ML
    
    Notebook d'exploration pour tester rapidement différents algorithmes.
    Utilise **gold.ml_datasets** depuis DuckLake.
    
    ⚠️ Pour tuning sérieux et MLflow tracking, voir `dagster_pipeline/` (Optuna + CatBoost).
    """)
    return


@app.cell
def _(get_client):
    client = get_client()
    conn = client.conn
    return client, conn


@app.cell
def _(conn, mo, pl):
    X_train = conn.execute("SELECT * FROM gold.X_train").pl()
    y_train = conn.execute("SELECT target FROM gold.y_train").pl().to_series().to_numpy()
    X_test = conn.execute("SELECT * FROM gold.X_test").pl()
    y_test = conn.execute("SELECT target FROM gold.y_test").pl().to_series().to_numpy()
    
    mo.md(f"""
    ## Datasets chargés
    
    - X_train: {X_train.shape}
    - y_train: {y_train.shape}
    - X_test: {X_test.shape}
    - y_test: {y_test.shape}
    """)
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_train):
    X_train.head(5)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Modèle 1 : Logistic Regression (baseline)
    """)
    return


@app.cell
def _(LogisticRegression, X_test, X_train, classification_report, confusion_matrix, mo, roc_auc_score, y_test, y_train):
    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr_model.fit(X_train.to_pandas(), y_train)
    
    y_pred_lr = lr_model.predict(X_test.to_pandas())
    y_proba_lr = lr_model.predict_proba(X_test.to_pandas())[:, 1]
    
    auc_lr = roc_auc_score(y_test, y_proba_lr)
    
    mo.md(f"""
    ### Résultats Logistic Regression
    
    **AUC-ROC** : {auc_lr:.4f}
    
    ```
    {classification_report(y_test, y_pred_lr)}
    ```
    
    **Matrice de confusion** :
    ```
    {confusion_matrix(y_test, y_pred_lr)}
    ```
    """)
    return auc_lr, lr_model, y_proba_lr, y_pred_lr


@app.cell
def _(mo):
    mo.md("""
    ## Modèle 2 : Random Forest
    """)
    return


@app.cell
def _(RandomForestClassifier, X_test, X_train, classification_report, confusion_matrix, mo, roc_auc_score, y_test, y_train):
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    rf_model.fit(X_train.to_pandas(), y_train)
    
    y_pred_rf = rf_model.predict(X_test.to_pandas())
    y_proba_rf = rf_model.predict_proba(X_test.to_pandas())[:, 1]
    
    auc_rf = roc_auc_score(y_test, y_proba_rf)
    
    mo.md(f"""
    ### Résultats Random Forest
    
    **AUC-ROC** : {auc_rf:.4f}
    
    ```
    {classification_report(y_test, y_pred_rf)}
    ```
    
    **Matrice de confusion** :
    ```
    {confusion_matrix(y_test, y_pred_rf)}
    ```
    """)
    return auc_rf, rf_model, y_proba_rf, y_pred_rf


@app.cell
def _(mo, np, pl, rf_model):
    feature_importance = pl.DataFrame({
        "feature": rf_model.feature_names_in_,
        "importance": rf_model.feature_importances_
    }).sort("importance", descending=True)
    
    mo.md("""
    ### Feature Importance (Random Forest)
    """)
    return (feature_importance,)


@app.cell
def _(feature_importance):
    feature_importance.head(15)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Comparaison
    """)
    return


@app.cell
def _(auc_lr, auc_rf, mo):
    mo.md(f"""
    | Modèle              | AUC-ROC |
    |---------------------|---------|
    | Logistic Regression | {auc_lr:.4f}  |
    | Random Forest       | {auc_rf:.4f}  |
    
    ✅ Meilleur : **{"Random Forest" if auc_rf > auc_lr else "Logistic Regression"}**
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## TODO
    
    - [ ] Tester CatBoost/LightGBM/XGBoost
    - [ ] Grid search hyperparamètres
    - [ ] Cross-validation
    - [ ] Analyse erreurs (FP, FN)
    - [ ] Sauvegarder modèle (joblib/S3)
    - [ ] Intégration MLflow tracking
    """)
    return


if __name__ == "__main__":
    app.run()
