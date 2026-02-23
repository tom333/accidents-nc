import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    import numpy as np
    from src.accidents.ducklake import get_client
    return HeatMap, MarkerCluster, folium, get_client, mo, np, pl


@app.cell
def _(mo):
    mo.md("""
    # Visualisations Prédictions
    
    Carte interactive des prédictions d'accidents avec heatmap et clusters.
    """)
    return


@app.cell
def _(get_client):
    client = get_client()
    conn = client.conn
    return client, conn


@app.cell
def _(mo):
    date_select = mo.ui.date(label="Date prédiction")
    hour_select = mo.ui.slider(0, 23, value=12, label="Heure", step=1)
    
    mo.hstack([date_select, hour_select])
    return date_select, hour_select


@app.cell
def _(conn, mo, pl):
    grid_points = conn.execute("""
        SELECT DISTINCT latitude, longitude
        FROM silver.full_dataset
        WHERE target = 0
        LIMIT 1000
    """).pl()
    
    mo.md(f"""
    ## Grille de prédiction
    
    **{len(grid_points):,}** points chargés (échantillon)
    """)
    return (grid_points,)


@app.cell
def _(grid_points):
    grid_points.head(10)
    return


@app.cell
def _(date_select, grid_points, hour_select, mo, np):
    predictions = grid_points.with_columns([
        pl.lit(np.random.rand(len(grid_points))).alias("risk_score")
    ])
    
    mo.md(f"""
    ## Prédictions générées
    
    Date : {date_select.value if date_select.value else "Non sélectionnée"}  
    Heure : {hour_select.value}h
    
    ⚠️ Scores aléatoires (TODO: intégrer vrai modèle ML)
    """)
    return (predictions,)


@app.cell
def _(predictions):
    predictions.sort("risk_score", descending=True).head(10)
    return


@app.cell
def _(HeatMap, folium, mo, np, predictions):
    center_lat = predictions["latitude"].mean()
    center_lon = predictions["longitude"].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="OpenStreetMap"
    )
    
    heat_data = predictions.select([
        "latitude",
        "longitude", 
        "risk_score"
    ]).to_numpy().tolist()
    
    HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(m)
    
    top_10 = predictions.sort("risk_score", descending=True).head(10)
    
    for row in top_10.iter_rows(named=True):
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8,
            color="red",
            fill=True,
            fillColor="red",
            fillOpacity=0.6,
            popup=f"Risque: {row['risk_score']:.3f}"
        ).add_to(m)
    
    mo.md("""
    ## Carte des risques
    """)
    return center_lat, center_lon, heat_data, m, top_10


@app.cell
def _(m, mo):
    mo.Html(m._repr_html_())
    return


@app.cell
def _(mo):
    mo.md("""
    ## TODO
    
    - [ ] Charger vrai modèle ML depuis S3
    - [ ] Calculer features temporelles dynamiques
    - [ ] Prédictions réelles (pas aléatoires)
    - [ ] Mode cluster (MarkerCluster)
    - [ ] Export GeoJSON
    - [ ] Statistiques par zone
    - [ ] Timeline animation (slider temporel)
    """)
    return


if __name__ == "__main__":
    app.run()
