import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import joblib
    import pandas as pd
    import numpy as np
    import geopandas as gpd
    from datetime import datetime, timedelta, date
    import folium
    from shapely.geometry import Point
    from folium.plugins import HeatMapWithTime
    from scipy.spatial.distance import cdist
    from geopy.distance import geodesic
    from sklearn.neighbors import NearestNeighbors
    return (
        NearestNeighbors,
        Point,
        cdist,
        date,
        folium,
        geodesic,
        gpd,
        joblib,
        mo,
        np,
        pd,
        timedelta,
    )


@app.cell
def _(mo):
    mo.md("""
    # 🗺️ Carte de Prédiction des Accidents

    Visualisez les zones à risque d'accident pour une date donnée, heure par heure.
    """)
    return


@app.cell
def _(Point, gpd, joblib, np, pd):


    # Chargement du modèle et des métadonnées
    model = joblib.load('accident_model.pkl')
    encoder = joblib.load('atm_encoder.pkl')
    features = joblib.load('features.pkl')

    # Charger les routes OSM
    routes_osm = gpd.read_file('routes.nc')

    # Créer une grille spatiale régulière (même config que l'entraînement)
    grid_step = 0.02  # ~2km
    buffer_meters = 200
    lat_min, lat_max = -23.0, -19.5
    lon_min, lon_max = 163.5, 168.0

    lats = np.arange(lat_min, lat_max, grid_step)
    lons = np.arange(lon_min, lon_max, grid_step)

    grid = pd.DataFrame([(lat, lon) for lat in lats for lon in lons], columns=["latitude", "longitude"])
    grid["geometry"] = grid.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
    grid_gdf = gpd.GeoDataFrame(grid, geometry="geometry", crs="EPSG:4326").to_crs(epsg=3857)

    # Buffer autour des routes
    routes_buffer = routes_osm.to_crs(epsg=3857).buffer(buffer_meters)
    routes_buffer_gdf = gpd.GeoDataFrame(geometry=routes_buffer, crs="EPSG:3857")

    # Spatial join : ne garder que les points de la grille qui sont sur/près des routes
    grid_on_roads = gpd.sjoin(grid_gdf, routes_buffer_gdf, how="inner", predicate="intersects").drop(columns="index_right")
    routes_grid = grid_on_roads.to_crs(epsg=4326).drop_duplicates(subset=['latitude', 'longitude'])

    # ==========================================
    # PRÉPARATION DES FEATURES OSM (road_type, speed_limit)
    # ==========================================
    road_types_mapping = {
        'motorway': 5, 'trunk': 4, 'primary': 3,
        'secondary': 2, 'tertiary': 2, 'residential': 1,
        'unclassified': 1, 'service': 1
    }

    routes_info = routes_osm.to_crs(epsg=4326).copy()
    if 'highway' in routes_info.columns:
        def get_road_type(highway):
            if isinstance(highway, np.ndarray):
                highway = highway[0] if len(highway) > 0 else 'unclassified'
            elif isinstance(highway, list):
                highway = highway[0] if len(highway) > 0 else 'unclassified'
            highway = str(highway) if highway is not None else 'unclassified'
            return road_types_mapping.get(highway, 1)

        routes_info['road_type_encoded'] = routes_info['highway'].apply(get_road_type)
        routes_info['speed_limit'] = routes_info.get('maxspeed', 50).fillna(50)
        if routes_info['speed_limit'].dtype == 'object':
            routes_info['speed_limit'] = routes_info['speed_limit'].apply(
                lambda x: int(str(x).split()[0]) if isinstance(x, str) and str(x).split()[0].isdigit() else 50
            )
    else:
        routes_info['road_type_encoded'] = 2
        routes_info['speed_limit'] = 50

    # Précomputer les centroids des routes
    route_centroids = routes_info.geometry.centroid
    route_coords = np.array([[p.y, p.x] for p in route_centroids])

    # Précomputer les features OSM pour chaque point de la grille
    grid_coords = routes_grid[['latitude', 'longitude']].values
    distances = cdist(grid_coords, route_coords)
    nearest_indices = distances.argmin(axis=1)

    routes_grid['road_type'] = routes_info.iloc[nearest_indices]['road_type_encoded'].values
    routes_grid['speed_limit'] = routes_info.iloc[nearest_indices]['speed_limit'].values

    print(f"✅ Modèle chargé : {type(model).__name__}")
    print(f"✅ Grille routière : {len(routes_grid)} points (échantillonnage {grid_step}° = ~{grid_step*111:.1f}km)")
    print(f"✅ Features : {len(features)} features enrichies")
    print(f"ℹ️  Total prédictions/heure : {len(routes_grid)} x 24h = {len(routes_grid)*24:,} points")
    return features, model, route_coords, routes_grid, routes_info


@app.cell
def _(date, mo, timedelta):
    # Interface utilisateur
    date_selector = mo.ui.date(
        value=date.today() + timedelta(days=1),
        label="📅 Date de prédiction",
        full_width=False
    )

    atm_selector = mo.ui.dropdown(
        options=['Normal', 'Pluie légère', 'Pluie forte', 'Brouillard'],
        value='Normal',
        label="🌦️ Conditions météorologiques"
    )

    mode_selector = mo.ui.radio(
        options=['top_n', 'threshold'],
        value='top_n',
        label="🎯 Mode de sélection"
    )

    top_n_slider = mo.ui.slider(
        start=1,
        stop=10,
        value=3,
        step=1,
        label="📊 Nombre de points à risque par heure (Top N)",
        full_width=True
    )

    threshold_slider = mo.ui.slider(
        start=0.5,
        stop=0.95,
        value=0.70,
        step=0.05,
        label="🎯 Seuil de probabilité minimale",
        full_width=True
    )

    mo.vstack([
        mo.hstack([date_selector, atm_selector]),
        mo.md("**Mode Top N** : Affiche les N points les plus dangereux par heure (recommandé)  \n**Mode Seuil** : Affiche tous les points au-dessus d'un seuil de probabilité"),
        mode_selector,
        top_n_slider #  if mode_selector.value == 'top_n' else threshold_slider
    ])
    return atm_selector, date_selector, threshold_slider


@app.cell
def _(
    NearestNeighbors,
    atm_selector,
    date_selector,
    features,
    geodesic,
    joblib,
    model,
    np,
    pd,
    routes_grid,
    threshold_slider,
):
    # Mapping conditions météo → codes
    atm_mapping = {
        'Normal': 1,
        'Pluie légère': 2,
        'Pluie forte': 3,
        'Brouillard': 5
    }

    selected_date = pd.to_datetime(date_selector.value)
    threshold = threshold_slider.value
    atm_code = atm_mapping.get(atm_selector.value, 1)

    # Extraire latitude et longitude depuis la géométrie
    grid_lats = routes_grid.geometry.y.values
    grid_lons = routes_grid.geometry.x.values

    # Charger les coordonnées des accidents pour la densité
    accidents_data = joblib.load('best_model_info.pkl')
    # Note: En production, il faudrait stocker les coordonnées des accidents séparément
    # Pour l'instant, on utilise une approximation avec les points de la grille
    # Dans un système réel, charger depuis une base de données

    # Centre de Nouméa
    noumea_center = (-22.2758, 166.4580)

    # Jours fériés Nouvelle-Calédonie
    nc_holidays_days = [
        (1, 1), (5, 1), (5, 8), (7, 14),
        (9, 24), (11, 1), (11, 11), (12, 25)
    ]

    # Générer prédictions pour chaque heure (0-23)
    predictions_by_hour = []

    for hour in range(24):
        # Créer DataFrame de base
        hourly_data = pd.DataFrame({
            'latitude': grid_lats,
            'longitude': grid_lons,
            'hour': hour,
            'dayofweek': selected_date.dayofweek,
            'month': selected_date.month,
            'atm': atm_code
        })

        # ==========================================
        # 1. FEATURES D'INTERACTION SPATIO-TEMPORELLES
        # ==========================================
        hourly_data['lat_hour'] = hourly_data['latitude'] * hourly_data['hour'] / 24
        hourly_data['lon_hour'] = hourly_data['longitude'] * hourly_data['hour'] / 24
        hourly_data['lat_dayofweek'] = hourly_data['latitude'] * hourly_data['dayofweek'] / 7
        hourly_data['lon_dayofweek'] = hourly_data['longitude'] * hourly_data['dayofweek'] / 7

        # Features temporelles binaires
        hourly_data['is_weekend'] = (hourly_data['dayofweek'] >= 5).astype(int)
        hourly_data['is_rush_morning'] = ((hourly_data['hour'] >= 7) & (hourly_data['hour'] <= 9)).astype(int)
        hourly_data['is_rush_evening'] = ((hourly_data['hour'] >= 17) & (hourly_data['hour'] <= 19)).astype(int)
        hourly_data['is_night'] = ((hourly_data['hour'] >= 22) | (hourly_data['hour'] <= 6)).astype(int)

        # Encodage cyclique
        hourly_data['hour_sin'] = np.sin(2 * np.pi * hourly_data['hour'] / 24)
        hourly_data['hour_cos'] = np.cos(2 * np.pi * hourly_data['hour'] / 24)
        hourly_data['dayofweek_sin'] = np.sin(2 * np.pi * hourly_data['dayofweek'] / 7)
        hourly_data['dayofweek_cos'] = np.cos(2 * np.pi * hourly_data['dayofweek'] / 7)

        # ==========================================
        # 2. FEATURES OSM (déjà précomputées dans routes_grid)
        # ==========================================
        hourly_data['road_type'] = routes_grid['road_type'].values
        hourly_data['speed_limit'] = routes_grid['speed_limit'].values

        # ==========================================
        # 3. FEATURES DE DENSITÉ ET PROXIMITÉ
        # ==========================================
        # Pour la démo, utiliser une approximation de densité constante
        # En production, charger depuis une base de données pré-calculée
        hourly_data['accident_density_5km'] = 2.0  # Valeur moyenne approximative

        # Distance à Nouméa
        hourly_data['dist_to_noumea_km'] = hourly_data.apply(
            lambda row: geodesic(
                (row['latitude'], row['longitude']),
                noumea_center
            ).km,
            axis=1
        )

        # ==========================================
        # 4. FEATURES TEMPORELLES AVANCÉES
        # ==========================================
        day = selected_date.day
        month_val = selected_date.month
        hourly_data['is_holiday'] = 1 if (month_val, day) in nc_holidays_days else 0
        hourly_data['school_holidays'] = 1 if month_val in [1, 7, 8, 12] else 0

        # Prédictions
        probas = model.predict_proba(hourly_data[features])[:, 1]

        # DEBUG: Voir la distribution des probabilités
        if hour == 0:
            print(f"\n🔍 DEBUG heure {hour}:")
            print(f"  Proba min: {probas.min():.6f}, max: {probas.max():.6f}, mean: {probas.mean():.6f}")
            print(f"  Proba > {threshold}: {(probas >= threshold).sum()} points")
            top_5_idx = probas.argsort()[-5:]
            print(f"  Top 5 probas: {probas[top_5_idx]}")

        # Filtrer par seuil
        mask = probas >= threshold
        if mask.sum() > 0:
            predictions_by_hour.append(pd.DataFrame({
                'latitude': hourly_data.loc[mask, 'latitude'].values,
                'longitude': hourly_data.loc[mask, 'longitude'].values,
                'hour': hour,
                'probability': probas[mask]
            }))

    # Concaténer toutes les heures
    if predictions_by_hour:
        all_predictions = pd.concat(predictions_by_hour, ignore_index=True)
    else:
        all_predictions = pd.DataFrame(columns=['latitude', 'longitude', 'hour', 'probability'])

    print(f"📊 Date : {selected_date.strftime('%Y-%m-%d')} ({selected_date.strftime('%A')})")
    print(f"🌦️ Conditions : {atm_selector.value}")
    print(f"🎯 Seuil : {threshold:.0%}")
    print(f"⚠️ Points à risque détectés : {len(all_predictions)}")

    all_predictions
    return (all_predictions,)


@app.cell
def _(all_predictions, folium, mo, routes_grid):
    # Créer la carte interactive

    # Centre de la carte (Nouvelle-Calédonie)
    map_center_lat = routes_grid.geometry.y.mean()
    map_center_lon = routes_grid.geometry.x.mean()

    folium_map = folium.Map(
        location=[map_center_lat, map_center_lon],
        zoom_start=9,
        tiles='OpenStreetMap'
    )

    # Ajouter les points par heure avec code couleur
    for map_hour in sorted(all_predictions['hour'].unique()):
        map_hour_data = all_predictions[all_predictions['hour'] == map_hour]

        # Groupe pour cette heure
        map_hour_group = folium.FeatureGroup(
            name=f"{map_hour:02d}h ({len(map_hour_data)} points)",
            show=map_hour < 24  # Afficher seulement les 3 premières heures par défaut
        )

        for _, row in map_hour_data.iterrows():
            # Couleur selon probabilité
            if row['probability'] >= 0.8:
                marker_color = 'red'
                marker_icon = 'exclamation-triangle'
            elif row['probability'] >= 0.6:
                marker_color = 'orange'
                marker_icon = 'warning'
            else:
                marker_color = 'yellow'
                marker_icon = 'info-sign'

            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"<b>{map_hour:02d}h</b><br>Risque: {row['probability']:.1%}",
                tooltip=f"{map_hour:02d}h - {row['probability']:.0%}",
                icon=folium.Icon(color=marker_color, icon=marker_icon, prefix='glyphicon')
            ).add_to(map_hour_group)

        map_hour_group.add_to(folium_map)

    # Contrôle des couches
    folium.LayerControl(collapsed=False).add_to(folium_map)

    # Légende
    # legend_html = f'''
    # <div style="position: fixed; 
    #             top: 10px; right: 10px; width: 250px; 
    #             background-color: white; border:2px solid grey; 
    #             z-index:9999; font-size:14px; padding: 10px">
    # <p><b>🗺️ Prédiction {selected_date.strftime('%d/%m/%Y')}</b></p>
    # <p><i class="glyphicon glyphicon-exclamation-triangle" style="color:red"></i> Risque élevé (≥80%)</p>
    # <p><i class="glyphicon glyphicon-warning-sign" style="color:orange"></i> Risque moyen (60-80%)</p>
    # <p><i class="glyphicon glyphicon-info-sign" style="color:gold"></i> Risque faible (seuil-60%)</p-->
    # <p><small>Utilisez le panneau pour filtrer par heure</small></p>
    # </div>
    # '''
    # folium_map.get_root().html.add_child(folium.Element(legend_html))

    mo.Html(folium_map._repr_html_())
    return


@app.cell
def _(all_predictions, mo):
    # Statistiques par heure

    stats_by_hour = all_predictions.groupby('hour').agg({
        'probability': ['count', 'mean', 'max']
    }).round(3)
    stats_by_hour.columns = ['Nombre de points', 'Risque moyen', 'Risque max']
    stats_by_hour = stats_by_hour.reset_index()
    stats_by_hour['hour'] = stats_by_hour['hour'].apply(lambda x: f"{x:02d}h")

    mo.md(f"""
    ## 📈 Statistiques par Heure

    Répartition des zones à risque sur 24 heures :
    """)

    mo.ui.table(stats_by_hour)
    return


@app.cell
def _(all_predictions, mo):
    # Résumé global

    top_hours = all_predictions.groupby('hour').size().nlargest(5)
    peak_hour = top_hours.index[0]
    peak_count = top_hours.values[0]
    avg_proba = all_predictions['probability'].mean()
    max_proba = all_predictions['probability'].max()

    mo.callout(
        f"""
        **🎯 Résumé de la Prédiction**

        - **Heure la plus dangereuse** : {peak_hour:02d}h ({peak_count} points à risque)
        - **Risque moyen** : {avg_proba:.1%}
        - **Risque maximum** : {max_proba:.1%}
        - **Total de points à risque** : {len(all_predictions)}
        """,
        kind="warn"
    )
    return


if __name__ == "__main__":
    app.run()
