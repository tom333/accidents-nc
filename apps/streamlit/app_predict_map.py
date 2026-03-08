#!/usr/bin/env python3
"""
Application Streamlit - Carte de Prédiction des Accidents en Nouvelle-Calédonie
Remplace le notebook predict_map.py avec une interface interactive complète
"""

import os
from datetime import date, datetime, timedelta

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import streamlit as st
from shapely.geometry import Point
from streamlit_folium import st_folium

# API URL - Devrait idéalement être dans un .env
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Configuration de la page
st.set_page_config(
    page_title="Prédiction des Accidents - NC",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
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
    max_value=date.today() + timedelta(days=30),
)

# Conditions météo
st.sidebar.markdown(
    "💡 *Sélectionnez les conditions météo prévues pour voir comment elles influencent les risques d'accident.*"
)
atm_options = {"Normal": 1, "Pluie légère": 2, "Pluie forte": 3, "Brouillard": 5}
selected_atm = st.sidebar.selectbox(
    "🌦️ Conditions météorologiques",
    list(atm_options.keys()),
    help="La météo influence fortement le risque d'accident. La pluie et le brouillard augmentent les dangers.",
)
atm_code = atm_options[selected_atm]

# Mode de sélection
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Mode d'Affichage")
st.sidebar.markdown("""💡 *Choisissez comment filtrer les zones affichées :*
- **Top N** : Montre uniquement les zones les plus dangereuses (recommandé pour une vue claire)
- **Seuil** : Montre toutes les zones au-dessus d'un niveau de risque
""")
display_mode = st.sidebar.radio("Sélectionner le mode", ["Top N par heure", "Seuil de probabilité"])

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
    help="Filtre les prédictions en dessous de ce seuil",
)
min_proba_filter = min_proba_percent / 100.0  # Conversion en décimal


# ==========================================
# FONCTION DE CHARGEMENT DES DONNÉES (CACHÉE)
# ==========================================
@st.cache_resource
def load_data():
    """Charge la grille et les routes (sans les modèles)."""
    with st.spinner("🔄 Préparation de la grille géographique..."):
        if not os.path.exists("routes.nc"):
            st.error("❌ Fichier routes.nc manquant.")
            st.stop()

        routes_osm = gpd.read_file("routes.nc")
        grid_step = 0.02
        buffer_meters = 200
        lat_min, lat_max = -23.0, -19.5
        lon_min, lon_max = 163.5, 168.0
        lats = np.arange(lat_min, lat_max, grid_step)
        lons = np.arange(lon_min, lon_max, grid_step)
        grid = pd.DataFrame(
            [(lat, lon) for lat in lats for lon in lons], columns=["latitude", "longitude"]
        )
        grid["geometry"] = grid.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
        grid_gdf = gpd.GeoDataFrame(grid, geometry="geometry", crs="EPSG:4326").to_crs(epsg=3857)
        routes_buffer = routes_osm.to_crs(epsg=3857).buffer(buffer_meters)
        routes_buffer_gdf = gpd.GeoDataFrame(geometry=routes_buffer, crs="EPSG:3857")
        grid_on_roads = gpd.sjoin(
            grid_gdf, routes_buffer_gdf, how="inner", predicate="intersects"
        ).drop(columns="index_right")
        routes_grid = grid_on_roads.to_crs(epsg=4326).drop_duplicates(
            subset=["latitude", "longitude"]
        )
        return routes_grid


# ==========================================
# FONCTION DE CALCUL DES FEATURES
# ==========================================
# Features are now calculated API-side.


# ==========================================
# FONCTION DE PRÉDICTION
# ==========================================
@st.cache_data
def generate_predictions(
    _routes_grid,
    selected_date,
    selected_atm,
    use_threshold,
    threshold_val=0.7,
    top_n_val=5,
):
    """Génère les prédictions en appelant l'API de batch"""

    predictions_by_hour = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Préparer les locations une seule fois
    locations = [
        {"latitude": float(row.geometry.y), "longitude": float(row.geometry.x)}
        for row in _routes_grid.itertuples()
    ]

    for hour in range(24):
        status_text.text(f"🔄 Appel API... {hour:02d}h00")
        progress_bar.progress((hour + 1) / 24)

        # Créer l'objet datetime combiné
        combined_dt = datetime.combine(selected_date, datetime.min.time()).replace(hour=hour)

        payload = {
            "locations": locations,
            "timestamp": combined_dt.isoformat(),
            "atm": selected_atm,
        }

        try:
            response = requests.post(f"{API_URL}/predict/batch", json=payload)
            response.raise_for_status()
            results = response.json()
            probas = np.array(results["predictions"])

            # Sélection selon le mode
            if use_threshold:
                mask = probas >= threshold_val
                if mask.sum() > 0:
                    predictions_by_hour.append(
                        pd.DataFrame(
                            {
                                "latitude": [
                                    loc["latitude"] for i, loc in enumerate(locations) if mask[i]
                                ],
                                "longitude": [
                                    loc["longitude"] for i, loc in enumerate(locations) if mask[i]
                                ],
                                "hour": hour,
                                "probability": probas[mask],
                            }
                        )
                    )
            else:
                top_indices = probas.argsort()[-top_n_val:][::-1]
                predictions_by_hour.append(
                    pd.DataFrame(
                        {
                            "latitude": [locations[i]["latitude"] for i in top_indices],
                            "longitude": [locations[i]["longitude"] for i in top_indices],
                            "hour": hour,
                            "probability": probas[top_indices],
                        }
                    )
                )
        except Exception as e:
            st.error(f"❌ Erreur lors de l'appel API à {hour}h : {e}")
            break

    progress_bar.empty()
    status_text.empty()

    if predictions_by_hour:
        return pd.concat(predictions_by_hour, ignore_index=True)
    return pd.DataFrame(columns=["latitude", "longitude", "hour", "probability"])


# ==========================================
# FONCTION DE CRÉATION DE LA CARTE
# ==========================================
def create_map(predictions, routes_grid):
    """Crée la carte Folium interactive"""

    # Centre de la carte
    map_center_lat = routes_grid.geometry.y.mean()
    map_center_lon = routes_grid.geometry.x.mean()

    m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=8, tiles="OpenStreetMap")

    # Ajouter tous les points directement sur la carte
    for _, row in predictions.iterrows():
        hour = int(row["hour"])

        # Couleur selon probabilité
        if row["probability"] >= 0.8:
            color = "red"
            icon = "exclamation-triangle"
            risk_level = "ÉLEVÉ"
        elif row["probability"] >= 0.6:
            color = "orange"
            icon = "warning"
            risk_level = "MOYEN"
        else:
            color = "yellow"
            icon = "info-sign"
            risk_level = "FAIBLE"

        # Popup détaillé
        popup_html = f"""
        <div style="font-family: Arial; width: 200px;">
            <h4 style="margin-bottom: 10px;">⚠️ Zone à Risque</h4>
            <p><b>Heure:</b> {hour:02d}h00</p>
            <p><b>Niveau:</b> <span style="color: {color};">{risk_level}</span></p>
            <p><b>Probabilité:</b> {row["probability"]:.1%}</p>
            <p><b>Position:</b><br>
               Lat: {row["latitude"]:.4f}<br>
               Lon: {row["longitude"]:.4f}</p>
        </div>
        """

        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{hour:02d}h - Risque {row['probability']:.0%}",
            icon=folium.Icon(color=color, icon=icon, prefix="glyphicon"),
        ).add_to(m)

    # Légende
    legend_html = """
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
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ==========================================
# CHARGEMENT DES DONNÉES
# ==========================================
routes_grid = load_data()
st.sidebar.success("✅ Données géographiques chargées")
st.sidebar.info(f"📍 Grille: {len(routes_grid):,} points")
st.sidebar.info(f"🔗 Connecté à l'API: {API_URL}")

# ==========================================
# GÉNÉRATION DES PRÉDICTIONS
# ==========================================
with st.spinner("🔮 Appel de l'API d'inférence..."):
    if use_threshold:
        predictions = generate_predictions(
            routes_grid, selected_date, selected_atm, True, threshold
        )
    else:
        predictions = generate_predictions(
            routes_grid, selected_date, selected_atm, False, top_n_val=top_n
        )

    # Appliquer le filtre de probabilité minimale
    if len(predictions) > 0:
        predictions = predictions[predictions["probability"] >= min_proba_filter]

# ==========================================
# AFFICHAGE DES RÉSULTATS
# ==========================================
if len(predictions) == 0:
    st.warning(
        "⚠️ Aucun point à risque détecté avec les paramètres actuels. Essayez de diminuer le seuil ou d'augmenter le Top N."
    )
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
            help="Nombre total de zones identifiées comme dangereuses sur l'ensemble de la journée",
        )

    with col2:
        avg_proba = predictions["probability"].mean()
        st.metric(
            "📈 Risque Moyen",
            f"{avg_proba:.1%}",
            help="Niveau de risque moyen de toutes les zones identifiées. Plus ce chiffre est élevé, plus la journée est globalement risquée.",
        )

    with col3:
        max_proba = predictions["probability"].max()
        st.metric(
            "⚠️ Risque Maximum",
            f"{max_proba:.1%}",
            help="Niveau de risque de la zone la plus dangereuse de la journée",
        )

    with col4:
        top_hours = predictions.groupby("hour").size().nlargest(1)
        peak_hour = top_hours.index[0]
        st.metric(
            "🕐 Heure Critique",
            f"{peak_hour:02d}h00",
            help="Heure de la journée avec le plus grand nombre de zones à risque",
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

    stats_by_hour = (
        predictions.groupby("hour").agg({"probability": ["count", "mean", "max", "min"]}).round(3)
    )
    stats_by_hour.columns = ["Nb Points", "Risque Moyen", "Risque Max", "Risque Min"]
    stats_by_hour = stats_by_hour.reset_index()
    stats_by_hour["hour"] = stats_by_hour["hour"].apply(lambda x: f"{x:02d}h00")
    stats_by_hour["Risque Moyen"] = stats_by_hour["Risque Moyen"].apply(lambda x: f"{x:.1%}")
    stats_by_hour["Risque Max"] = stats_by_hour["Risque Max"].apply(lambda x: f"{x:.1%}")
    stats_by_hour["Risque Min"] = stats_by_hour["Risque Min"].apply(lambda x: f"{x:.1%}")

    st.dataframe(stats_by_hour, use_container_width=True)

    # Distribution des risques
    st.markdown("---")
    st.subheader("📈 Analyse Approfondie")
    st.markdown("""Cette section vous permet de mieux comprendre la **répartition des niveaux de risque**
et d'identifier les **zones les plus critiques** où concentrer votre attention.""")

    col1, col2 = st.columns(2)

    with col1:
        # Histogramme
        hist_data = predictions["probability"]
        st.bar_chart(hist_data.value_counts(bins=10).sort_index())

    with col2:
        # Top 10 zones les plus dangereuses
        st.markdown("**🎯 Top 10 Zones les Plus Dangereuses**")
        top_10 = predictions.nlargest(10, "probability")[
            ["hour", "latitude", "longitude", "probability"]
        ]
        top_10["hour"] = top_10["hour"].apply(lambda x: f"{x:02d}h00")
        top_10["probability"] = top_10["probability"].apply(lambda x: f"{x:.1%}")
        st.dataframe(top_10, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; font-size: 12px;'>
    <p>🔮 Prédictions via API Gateway (Blending CatBoost/XGBoost/MLP)</p>
    <p>📊 {len(FEATURE_COLUMNS)} features enrichies | 🗺️ Basé sur OpenStreetMap</p>
    <p>⚠️ Ces prédictions sont indicatives et ne remplacent pas la prudence au volant</p>
</div>
""",
    unsafe_allow_html=True,
)
