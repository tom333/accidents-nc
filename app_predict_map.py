#!/usr/bin/env python3
"""
Application Streamlit - Carte de Prédiction des Accidents en Nouvelle-Calédonie
Remplace le notebook predict_map.py avec une interface interactive complète
"""
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta, date
from pathlib import Path
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point
from scipy.spatial.distance import cdist
from geopy.distance import geodesic

# Configuration de la page
st.set_page_config(
    page_title="Prédiction des Accidents - NC",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🗺️ Carte de Prédiction des Accidents - Nouvelle-Calédonie")
st.markdown("""
### ℹ️ Comment ça marche ?

Cette application utilise l'**intelligence artificielle** pour prédire où les accidents de la route sont les plus susceptibles de se produire en Nouvelle-Calédonie. 
En analysant des années de données d'accidents passés, les caractéristiques des routes, et les conditions météo, notre modèle identifie les **zones et moments les plus à risque**.

💡 **Utilisez ces prédictions pour** :
- Identifier les zones dangereuses à éviter
- Planifier vos trajets aux heures les moins risquées
- Adapter votre conduite selon les conditions météo

⚠️ **Important** : Ces prédictions sont des **estimations statistiques**. Elles ne peuvent pas prédire où un accident va réellement se produire, mais indiquent les zones historiquement plus risquées. Restez toujours prudent sur la route !
""")

st.markdown("---")

# ==========================================
# SIDEBAR - PARAMÈTRES
# ==========================================
st.sidebar.header("⚙️ Paramètres de Prédiction")

# Sélection de la date
selected_date = st.sidebar.date_input(
    "📅 Date de prédiction",
    value=date.today() + timedelta(days=1),
    min_value=date.today(),
    max_value=date.today() + timedelta(days=30)
)

# Conditions météo
st.sidebar.markdown("💡 *Sélectionnez les conditions météo prévues pour voir comment elles influencent les risques d'accident.*")
atm_options = {
    'Normal': 1,
    'Pluie légère': 2,
    'Pluie forte': 3,
    'Brouillard': 5
}
selected_atm = st.sidebar.selectbox(
    "🌦️ Conditions météorologiques", 
    list(atm_options.keys()),
    help="La météo influence fortement le risque d'accident. La pluie et le brouillard augmentent les dangers."
)
atm_code = atm_options[selected_atm]

# Mode de sélection
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Mode d'Affichage")
st.sidebar.markdown("""💡 *Choisissez comment filtrer les zones affichées :*
- **Top N** : Montre uniquement les zones les plus dangereuses (recommandé pour une vue claire)
- **Seuil** : Montre toutes les zones au-dessus d'un niveau de risque
""")
display_mode = st.sidebar.radio(
    "Sélectionner le mode",
    ["Top N par heure", "Seuil de probabilité"]
)

if display_mode == "Top N par heure":
    top_n = st.sidebar.slider("📊 Nombre de points à risque par heure", 1, 20, 5)
    use_threshold = False
else:
    threshold = st.sidebar.slider("🎯 Seuil de probabilité minimale", 0.5, 0.95, 0.70, 0.05)
    use_threshold = True

# Options d'affichage
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Options d'Affichage")
min_proba_percent = st.sidebar.slider(
    "🎯 Probabilité minimale à afficher",
    min_value=0,
    max_value=100,
    value=80,
    step=5,
    format="%d%%",
    help="Filtre les prédictions en dessous de ce seuil"
)
min_proba_filter = min_proba_percent / 100.0  # Conversion en décimal

# ==========================================
# FONCTION DE CHARGEMENT DES DONNÉES (CACHÉE)
# ==========================================
@st.cache_resource
def load_model_and_data():
    """Charge le modèle, les routes et les métadonnées"""
    with st.spinner("🔄 Chargement du modèle et des données..."):
        # Charger le modèle
        model = joblib.load('accident_model.pkl')
        encoder = joblib.load('atm_encoder.pkl')
        features = joblib.load('features.pkl')
        
        routes_grid = None
        routes_features_path = Path("routes_with_features.pkl")

        if routes_features_path.exists():
            routes_data = joblib.load(routes_features_path)

            if isinstance(routes_data, dict) and 'routes_grid' in routes_data:
                routes_data = routes_data['routes_grid']

            routes_grid = ensure_geodataframe(routes_data)
            routes_grid = routes_grid.drop_duplicates(subset=['latitude', 'longitude']).reset_index(drop=True)
        else:
            st.warning(
                "⚠️ routes_with_features.pkl introuvable : recalcul des features OSM (moins précis)."
            )

            routes_osm = gpd.read_file('routes.nc')

            grid_step = 0.02  # ~2km
            buffer_meters = 200
            lat_min, lat_max = -23.0, -19.5
            lon_min, lon_max = 163.5, 168.0

            lats = np.arange(lat_min, lat_max, grid_step)
            lons = np.arange(lon_min, lon_max, grid_step)

            grid = pd.DataFrame(
                [(lat, lon) for lat in lats for lon in lons],
                columns=["latitude", "longitude"]
            )
            grid["geometry"] = grid.apply(
                lambda row: Point(row["longitude"], row["latitude"]), axis=1
            )
            grid_gdf = gpd.GeoDataFrame(grid, geometry="geometry", crs="EPSG:4326").to_crs(epsg=3857)

            routes_buffer = routes_osm.to_crs(epsg=3857).buffer(buffer_meters)
            routes_buffer_gdf = gpd.GeoDataFrame(geometry=routes_buffer, crs="EPSG:3857")

            grid_on_roads = gpd.sjoin(
                grid_gdf, routes_buffer_gdf, how="inner", predicate="intersects"
            ).drop(columns="index_right")
            routes_grid = grid_on_roads.to_crs(epsg=4326)

            road_types_mapping = {
                'motorway': 5, 'trunk': 4, 'primary': 3,
                'secondary': 2, 'tertiary': 2, 'residential': 1,
                'unclassified': 1, 'service': 1
            }

            routes_info = routes_osm.to_crs(epsg=4326).copy()
            if 'highway' in routes_info.columns:
                def get_road_type(highway):
                    if isinstance(highway, (np.ndarray, list)):
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

            route_centroids = routes_info.geometry.centroid
            route_coords = np.array([[p.y, p.x] for p in route_centroids])
            grid_coords = routes_grid[['latitude', 'longitude']].values

            distances = cdist(grid_coords, route_coords)
            nearest_indices = distances.argmin(axis=1)

            routes_grid['road_type'] = routes_info.iloc[nearest_indices]['road_type_encoded'].values
            routes_grid['speed_limit'] = routes_info.iloc[nearest_indices]['speed_limit'].values

            routes_grid = ensure_geodataframe(routes_grid)
            routes_grid = routes_grid.drop_duplicates(subset=['latitude', 'longitude']).reset_index(drop=True)

        return model, features, routes_grid, encoder


def encode_atm_value(encoder, atm_value):
    """Mappe la condition météo brute vers la valeur attendue par le modèle."""
    if encoder is None:
        return atm_value
    try:
        return int(encoder.transform([atm_value])[0])
    except (ValueError, AttributeError):
        st.warning(
            "🌦️ Condition météo inconnue de l'encodeur, utilisation de la valeur brute."
        )
        return atm_value


def ensure_geodataframe(data):
    """Garantit un GeoDataFrame sur backend numpy pour éviter les soucis pyarrow."""
    if isinstance(data, gpd.GeoDataFrame):
        base = data.copy()
        if base.crs is None:
            base = base.set_crs(epsg=4326)
    else:
        base = pd.DataFrame(data).copy()
        if 'geometry' in base.columns:
            geom_values = base['geometry']
        else:
            geom_values = gpd.points_from_xy(base['longitude'], base['latitude'])
        base = gpd.GeoDataFrame(
            base.drop(columns='geometry', errors='ignore'),
            geometry=geom_values,
            crs="EPSG:4326"
        )

    columns_numpy = {}
    for col in base.columns:
        if col == 'geometry':
            columns_numpy[col] = np.array(base.geometry)
        else:
            series = base[col]
            if hasattr(series, 'to_numpy'):
                columns_numpy[col] = series.to_numpy()
            else:
                columns_numpy[col] = series.values

    return gpd.GeoDataFrame(columns_numpy, geometry='geometry', crs=base.crs)

# ==========================================
# FONCTION DE CALCUL DES FEATURES
# ==========================================
def calculate_features(routes_grid, selected_date, atm_value, hour):
    """Calcule toutes les features enrichies pour une heure donnée."""
    
    # Extraire coordonnées depuis la géométrie
    grid_lats = routes_grid.geometry.y.values
    grid_lons = routes_grid.geometry.x.values
    
    # DataFrame de base
    hourly_data = pd.DataFrame({
        'latitude': grid_lats,
        'longitude': grid_lons,
        'hour': hour,
        'dayofweek': selected_date.weekday(),
        'month': selected_date.month,
        'atm': atm_value
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
    hourly_data['is_rush_morning'] = ((hourly_data['hour'] >= 6) & (hourly_data['hour'] <= 8)).astype(int)
    hourly_data['is_rush_evening'] = ((hourly_data['hour'] >= 16) & (hourly_data['hour'] <= 18)).astype(int)
    hourly_data['is_night'] = ((hourly_data['hour'] >= 19) | (hourly_data['hour'] <= 5)).astype(int)
    
    # Encodage cyclique (éviter discontinuité 23h→0h)
    hourly_data['hour_sin'] = np.sin(2 * np.pi * hourly_data['hour'] / 24)
    hourly_data['hour_cos'] = np.cos(2 * np.pi * hourly_data['hour'] / 24)
    hourly_data['dayofweek_sin'] = np.sin(2 * np.pi * hourly_data['dayofweek'] / 7)
    hourly_data['dayofweek_cos'] = np.cos(2 * np.pi * hourly_data['dayofweek'] / 7)
    
    # ==========================================
    # 2. FEATURES OSM (déjà dans routes_grid)
    # ==========================================
    hourly_data['road_type'] = routes_grid['road_type'].values
    hourly_data['speed_limit'] = routes_grid['speed_limit'].values
    
    # ==========================================
    # 3. FEATURES DE DENSITÉ ET PROXIMITÉ
    # ==========================================
    if 'accident_density_5km' in routes_grid.columns:
        hourly_data['accident_density_5km'] = routes_grid['accident_density_5km'].values
    else:
        hourly_data['accident_density_5km'] = 2.0

    if 'dist_to_noumea_km' in routes_grid.columns:
        hourly_data['dist_to_noumea_km'] = routes_grid['dist_to_noumea_km'].values
    else:
        noumea_center = (-22.2758, 166.4580)
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
    nc_holidays_days = [
        (1, 1), (5, 1), (5, 8), (7, 14),
        (9, 24), (11, 1), (11, 11), (12, 25)
    ]
    
    day = selected_date.day
    month_val = selected_date.month
    hourly_data['is_holiday'] = 1 if (month_val, day) in nc_holidays_days else 0
    hourly_data['school_holidays'] = 1 if month_val in [1, 7, 8, 12] else 0
    
    return hourly_data

# ==========================================
# FONCTION DE PRÉDICTION
# ==========================================
@st.cache_data
def generate_predictions(_model, _routes_grid, features, selected_date, atm_value, use_threshold, threshold_val=0.7, top_n_val=5):
    """Génère les prédictions pour toutes les heures"""
    
    predictions_by_hour = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for hour in range(24):
        status_text.text(f"🔄 Calcul en cours... {hour+1}/24 heures")
        progress_bar.progress((hour + 1) / 24)
        
        # Calculer les features pour cette heure
        hourly_data = calculate_features(_routes_grid, selected_date, atm_value, hour)
        
        # Prédictions
        probas = _model.predict_proba(hourly_data[features])[:, 1]
        
        # Sélection selon le mode
        if use_threshold:
            mask = probas >= threshold_val
            if mask.sum() > 0:
                predictions_by_hour.append(pd.DataFrame({
                    'latitude': hourly_data.loc[mask, 'latitude'].values,
                    'longitude': hourly_data.loc[mask, 'longitude'].values,
                    'hour': hour,
                    'probability': probas[mask]
                }))
        else:
            # Top N par heure
            top_indices = probas.argsort()[-top_n_val:][::-1]
            predictions_by_hour.append(pd.DataFrame({
                'latitude': hourly_data.iloc[top_indices]['latitude'].values,
                'longitude': hourly_data.iloc[top_indices]['longitude'].values,
                'hour': hour,
                'probability': probas[top_indices]
            }))
    
    progress_bar.empty()
    status_text.empty()
    
    # Concaténer toutes les heures
    if predictions_by_hour:
        all_predictions = pd.concat(predictions_by_hour, ignore_index=True)
    else:
        all_predictions = pd.DataFrame(columns=['latitude', 'longitude', 'hour', 'probability'])
    
    return all_predictions

# ==========================================
# FONCTION DE CRÉATION DE LA CARTE
# ==========================================
def create_map(predictions, routes_grid):
    """Crée la carte Folium interactive"""
    
    # Centre de la carte
    map_center_lat = routes_grid.geometry.y.mean()
    map_center_lon = routes_grid.geometry.x.mean()
    
    m = folium.Map(
        location=[map_center_lat, map_center_lon],
        zoom_start=8,
        tiles='OpenStreetMap'
    )
    
    # Ajouter tous les points directement sur la carte
    for _, row in predictions.iterrows():
        hour = int(row['hour'])
        
        # Couleur selon probabilité
        if row['probability'] >= 0.8:
            color = 'red'
            icon = 'exclamation-triangle'
            risk_level = "ÉLEVÉ"
        elif row['probability'] >= 0.6:
            color = 'orange'
            icon = 'warning'
            risk_level = "MOYEN"
        else:
            color = 'yellow'
            icon = 'info-sign'
            risk_level = "FAIBLE"
        
        # Popup détaillé
        popup_html = f"""
        <div style="font-family: Arial; width: 200px;">
            <h4 style="margin-bottom: 10px;">⚠️ Zone à Risque</h4>
            <p><b>Heure:</b> {hour:02d}h00</p>
            <p><b>Niveau:</b> <span style="color: {color};">{risk_level}</span></p>
            <p><b>Probabilité:</b> {row['probability']:.1%}</p>
            <p><b>Position:</b><br>
               Lat: {row['latitude']:.4f}<br>
               Lon: {row['longitude']:.4f}</p>
        </div>
        """
        
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{hour:02d}h - Risque {row['probability']:.0%}",
            icon=folium.Icon(color=color, icon=icon, prefix='glyphicon')
        ).add_to(m)
    
    # Légende
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; right: 10px; width: 280px; 
                background-color: white; border:2px solid grey; 
                z-index:9999; font-size:13px; padding: 15px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <h4 style="margin-top: 0;">🗺️ Légende</h4>
        <p style="margin: 5px 0;">
            <i class="fa fa-circle" style="color:red"></i> 
            Risque ÉLEVÉ (≥80%)
        </p>
        <p style="margin: 5px 0;">
            <i class="fa fa-circle" style="color:orange"></i> 
            Risque MOYEN (60-80%)
        </p>
        <p style="margin: 5px 0;">
            <i class="fa fa-circle" style="color:gold"></i> 
            Risque FAIBLE (&lt;60%)
        </p>
        <hr style="margin: 10px 0;">
        <p style="font-size: 11px; color: #666; margin: 0;">
            💡 Utilisez le panneau de couches pour filtrer par heure
        </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

# ==========================================
# CHARGEMENT DES DONNÉES
# ==========================================
model, features, routes_grid, atm_encoder = load_model_and_data()

atm_encoded_value = encode_atm_value(atm_encoder, atm_code)

st.sidebar.success(f"✅ Modèle chargé: {type(model).__name__}")
st.sidebar.info(f"📍 Grille: {len(routes_grid):,} points")

# ==========================================
# GÉNÉRATION DES PRÉDICTIONS
# ==========================================
with st.spinner("🔮 Génération des prédictions..."):
    if use_threshold:
        predictions = generate_predictions(
            model, routes_grid, features, 
            pd.to_datetime(selected_date), atm_encoded_value,
            True, threshold
        )
    else:
        predictions = generate_predictions(
            model, routes_grid, features,
            pd.to_datetime(selected_date), atm_encoded_value,
            False, top_n_val=top_n
        )
    
    # Appliquer le filtre de probabilité minimale
    if len(predictions) > 0:
        predictions = predictions[predictions['probability'] >= min_proba_filter]

# ==========================================
# AFFICHAGE DES RÉSULTATS
# ==========================================
if len(predictions) == 0:
    st.warning("⚠️ Aucun point à risque détecté avec les paramètres actuels. Essayez de diminuer le seuil ou d'augmenter le Top N.")
else:
    # Explication des résultats
    st.markdown("### 📊 Résumé des Prédictions")
    st.markdown("""Voici un aperçu global des zones à risque identifiées pour la journée sélectionnée. 
Les **probabilités** représentent le niveau de risque : plus le pourcentage est élevé, plus le risque d'accident est important dans cette zone.""")
    
    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📊 Points à Risque", 
            f"{len(predictions):,}",
            help="Nombre total de zones identifiées comme dangereuses sur l'ensemble de la journée"
        )
    
    with col2:
        avg_proba = predictions['probability'].mean()
        st.metric(
            "📈 Risque Moyen", 
            f"{avg_proba:.1%}",
            help="Niveau de risque moyen de toutes les zones identifiées. Plus ce chiffre est élevé, plus la journée est globalement risquée."
        )
    
    with col3:
        max_proba = predictions['probability'].max()
        st.metric(
            "⚠️ Risque Maximum", 
            f"{max_proba:.1%}",
            help="Niveau de risque de la zone la plus dangereuse de la journée"
        )
    
    with col4:
        top_hours = predictions.groupby('hour').size().nlargest(1)
        peak_hour = top_hours.index[0]
        st.metric(
            "🕐 Heure Critique", 
            f"{peak_hour:02d}h00",
            help="Heure de la journée avec le plus grand nombre de zones à risque"
        )
    
    # Carte interactive
    st.markdown("---")
    st.subheader("🗺️ Carte Interactive des Zones à Risque")
    st.markdown("""**Comment lire la carte :**
    
- 🔴 **Marqueur rouge** : Zone à risque **ÉLEVÉ** (≥80%) - Soyez particulièrement vigilant
- 🟠 **Marqueur orange** : Zone à risque **MOYEN** (60-80%) - Prudence recommandée  
- 🟡 **Marqueur jaune** : Zone à risque **FAIBLE** (<60%) - Attention de base

💡 **Astuce** : Cliquez sur un marqueur pour voir les détails (heure exacte, niveau de risque, coordonnées GPS).
""")
    
    map_obj = create_map(predictions, routes_grid)
    st_folium(map_obj, width=1400, height=600)
    
    # Statistiques détaillées
    st.markdown("---")
    st.subheader("📊 Statistiques Détaillées par Heure")
    st.markdown("""Ce tableau montre l'évolution du risque d'accident **heure par heure** tout au long de la journée. 
Cela vous permet d'identifier les **créneaux horaires les plus dangereux** pour planifier vos déplacements.
""")
    
    stats_by_hour = predictions.groupby('hour').agg({
        'probability': ['count', 'mean', 'max', 'min']
    }).round(3)
    stats_by_hour.columns = ['Nb Points', 'Risque Moyen', 'Risque Max', 'Risque Min']
    stats_by_hour = stats_by_hour.reset_index()
    stats_by_hour['hour'] = stats_by_hour['hour'].apply(lambda x: f"{x:02d}h00")
    stats_by_hour['Risque Moyen'] = stats_by_hour['Risque Moyen'].apply(lambda x: f"{x:.1%}")
    stats_by_hour['Risque Max'] = stats_by_hour['Risque Max'].apply(lambda x: f"{x:.1%}")
    stats_by_hour['Risque Min'] = stats_by_hour['Risque Min'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(stats_by_hour, use_container_width=True)
    
    # Distribution des risques
    st.markdown("---")
    st.subheader("📈 Analyse Approfondie")
    st.markdown("""Cette section vous permet de mieux comprendre la **répartition des niveaux de risque** 
et d'identifier les **zones les plus critiques** où concentrer votre attention.""")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogramme
        hist_data = predictions['probability']
        st.bar_chart(hist_data.value_counts(bins=10).sort_index())
    
    with col2:
        # Top 10 zones les plus dangereuses
        st.markdown("**🎯 Top 10 Zones les Plus Dangereuses**")
        top_10 = predictions.nlargest(10, 'probability')[['hour', 'latitude', 'longitude', 'probability']]
        top_10['hour'] = top_10['hour'].apply(lambda x: f"{x:02d}h00")
        top_10['probability'] = top_10['probability'].apply(lambda x: f"{x:.1%}")
        st.dataframe(top_10, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    <p>🔮 Prédictions générées par modèle ML optimisé (CatBoost/LightGBM/XGBoost)</p>
    <p>📊 {features_count} features enrichies | 🗺️ Basé sur OpenStreetMap</p>
    <p>⚠️ Ces prédictions sont indicatives et ne remplacent pas la prudence au volant</p>
</div>
""".format(features_count=len(features)), unsafe_allow_html=True)
