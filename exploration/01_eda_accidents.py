import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    from src.accidents.ducklake import get_client
    return alt, get_client, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # EDA - Accidents Nouvelle-Calédonie
    
    Analyse exploratoire des données d'accidents routiers (dep 988).
    Données chargées depuis **DuckLake** (bronze.caracteristiques).
    """)
    return


@app.cell
def _(get_client):
    client = get_client()
    conn = client.conn
    return client, conn


@app.cell
def _(conn, mo, pl):
    accidents_nc = conn.execute("""
        SELECT *
        FROM bronze.caracteristiques
        WHERE dep = '988'
    """).pl()
    
    mo.md(f"""
    ## Données brutes
    
    **{len(accidents_nc):,}** accidents enregistrés en Nouvelle-Calédonie
    """)
    return (accidents_nc,)


@app.cell
def _(accidents_nc):
    accidents_nc.head(10)
    return


@app.cell
def _(accidents_nc, mo, pl):
    stats_temporelles = accidents_nc.group_by("an").agg([
        pl.count().alias("nb_accidents"),
        pl.col("Num_Acc").n_unique().alias("nb_uniques")
    ]).sort("an")
    
    mo.md("""
    ## Distribution temporelle
    """)
    return (stats_temporelles,)


@app.cell
def _(alt, stats_temporelles):
    chart_annees = alt.Chart(stats_temporelles).mark_bar().encode(
        x=alt.X("an:O", title="Année"),
        y=alt.Y("nb_accidents:Q", title="Nombre d'accidents"),
        tooltip=["an", "nb_accidents"]
    ).properties(
        title="Accidents par année",
        width=600,
        height=300
    )
    chart_annees
    return (chart_annees,)


@app.cell
def _(accidents_nc, mo, pl):
    stats_mois = accidents_nc.group_by("mois").agg([
        pl.count().alias("nb_accidents")
    ]).sort("mois")
    
    mo.md("""
    ## Saisonnalité
    """)
    return (stats_mois,)


@app.cell
def _(alt, stats_mois):
    chart_mois = alt.Chart(stats_mois).mark_line(point=True).encode(
        x=alt.X("mois:O", title="Mois"),
        y=alt.Y("nb_accidents:Q", title="Nombre d'accidents"),
        tooltip=["mois", "nb_accidents"]
    ).properties(
        title="Accidents par mois",
        width=600,
        height=300
    )
    chart_mois
    return (chart_mois,)


@app.cell
def _(accidents_nc, mo):
    stats_atm = accidents_nc.group_by("atm").agg([
        pl.count().alias("nb_accidents")
    ]).sort("nb_accidents", descending=True)
    
    mo.md("""
    ## Conditions atmosphériques
    """)
    return (stats_atm,)


@app.cell
def _(stats_atm):
    stats_atm
    return


@app.cell
def _(accidents_nc, mo):
    coords_valides = accidents_nc.filter(
        (pl.col("lat").is_not_null()) & (pl.col("long").is_not_null())
    )
    
    mo.md(f"""
    ## Géolocalisation
    
    **{len(coords_valides):,}** accidents géolocalisés ({len(coords_valides)/len(accidents_nc)*100:.1f}%)
    """)
    return (coords_valides,)


@app.cell
def _(coords_valides):
    coords_valides.select(["lat", "long", "com", "agg"]).head(10)
    return


@app.cell
def _(mo):
    mo.md("""
    ## TODO
    
    - [ ] Carte interactive avec Folium
    - [ ] Analyse gravité (usagers)
    - [ ] Corrélations temporelles avancées
    - [ ] Analyse par commune
    """)
    return


if __name__ == "__main__":
    app.run()
