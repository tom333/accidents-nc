import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    from src.accidents.ducklake import get_client
    from src.accidents.utils.temporal import add_temporal_features
    return add_temporal_features, alt, get_client, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # Feature Engineering - Exploration
    
    Test et visualisation de nouvelles features pour améliorer les prédictions.
    Données depuis **silver.full_dataset** (DuckLake).
    """)
    return


@app.cell
def _(get_client):
    client = get_client()
    conn = client.conn
    return client, conn


@app.cell
def _(conn, mo, pl):
    full_dataset = conn.execute("""
        SELECT *
        FROM silver.full_dataset
        LIMIT 10000
    """).pl()
    
    mo.md(f"""
    ## Dataset complet (échantillon)
    
    **{len(full_dataset):,}** observations chargées  
    Target balance : {full_dataset.filter(pl.col("target") == 1).shape[0]} accidents / {full_dataset.filter(pl.col("target") == 0).shape[0]} non-accidents
    """)
    return (full_dataset,)


@app.cell
def _(full_dataset):
    full_dataset.head(5)
    return


@app.cell
def _(full_dataset, mo):
    mo.md("""
    ## Features temporelles existantes
    """)
    return


@app.cell
def _(full_dataset):
    temporal_cols = [col for col in full_dataset.columns if any(x in col for x in ["hour", "day", "month", "sin", "cos", "weekend", "rush"])]
    full_dataset.select(["accident_datetime"] + temporal_cols).head(10)
    return (temporal_cols,)


@app.cell
def _(alt, full_dataset, pl):
    accidents_par_heure = full_dataset.filter(pl.col("target") == 1).group_by("hour_of_day").agg([
        pl.count().alias("nb_accidents")
    ]).sort("hour_of_day")
    
    chart_heures = alt.Chart(accidents_par_heure).mark_bar().encode(
        x=alt.X("hour_of_day:O", title="Heure"),
        y=alt.Y("nb_accidents:Q", title="Nombre d'accidents"),
        tooltip=["hour_of_day", "nb_accidents"]
    ).properties(
        title="Distribution horaire des accidents",
        width=700,
        height=300
    )
    chart_heures
    return accidents_par_heure, chart_heures


@app.cell
def _(mo):
    mo.md("""
    ## Idées de nouvelles features
    
    ### 1. Features cycliques améliorées
    - Heure dans la journée (sin/cos déjà présent)
    - Jour dans la semaine
    - Semaine dans l'année
    
    ### 2. Features d'interaction
    - weekend × rush_hour
    - holiday × hour
    - atm × hour (conditions météo selon moment)
    
    ### 3. Features géospatiales avancées
    - Distance au centre-ville
    - Proximité zone commerciale
    - Densité routière locale
    """)
    return


@app.cell
def _(full_dataset, pl):
    features_interactions = full_dataset.with_columns([
        (pl.col("is_weekend") * pl.col("is_rush_hour")).alias("weekend_rush"),
        (pl.col("is_holiday") * pl.col("hour_of_day")).alias("holiday_hour"),
    ])
    
    features_interactions.select(["is_weekend", "is_rush_hour", "weekend_rush", "is_holiday", "hour_of_day", "holiday_hour"]).head(10)
    return (features_interactions,)


@app.cell
def _(alt, features_interactions, pl):
    correlation_weekend_rush = features_interactions.filter(pl.col("target") == 1).group_by("weekend_rush").agg([
        pl.count().alias("nb_accidents")
    ])
    
    chart_weekend_rush = alt.Chart(correlation_weekend_rush).mark_bar().encode(
        x=alt.X("weekend_rush:O", title="Weekend × Rush Hour"),
        y=alt.Y("nb_accidents:Q", title="Accidents"),
        tooltip=["weekend_rush", "nb_accidents"]
    ).properties(
        title="Accidents selon weekend + rush hour",
        width=400,
        height=250
    )
    chart_weekend_rush
    return chart_weekend_rush, correlation_weekend_rush


@app.cell
def _(mo):
    mo.md("""
    ## TODO - Expérimentations
    
    - [ ] Tester importance features avec RandomForest
    - [ ] Feature selection (SHAP, permutation importance)
    - [ ] Encoder les variables catégorielles (atm, col, etc.)
    - [ ] Normalisation/standardisation
    - [ ] PCA pour réduction dimensionnalité
    """)
    return


if __name__ == "__main__":
    app.run()
