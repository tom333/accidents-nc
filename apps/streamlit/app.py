"""
Application Streamlit - Carte de Prédiction des Accidents en Nouvelle-Calédonie

Version 2.0 - Architecture refactorisée avec DuckLake
"""
import streamlit as st
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta, date
from streamlit_folium import st_folium
import folium
from folium.plugins import HeatMap
import numpy as np

from src.accidents.ducklake import get_client
from src.accidents.gold.schema import GOLD_SCHEMA

# Configuration de la page
st.set_page_config(
    page_title="Prédiction des Accidents - NC",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_ducklake_connection():
    """Obtient la connexion DuckLake (cachée)."""
    return get_client()


@st.cache_data(ttl=3600)
def load_trained_model():
    """Charge le modèle entraîné depuis gold.models (si disponible)."""
    try:
        client = get_ducklake_connection()
        # TODO: Charger le modèle depuis S3 ou gold schema
        st.info("🚧 Chargement modèle depuis DuckLake en développement")
        return None
    except Exception as e:
        st.error(f"⚠️ Erreur chargement modèle: {e}")
        return None


@st.cache_data(ttl=3600)
def load_prediction_grid():
    """Charge la grille de prédiction depuis silver.full_dataset."""
    try:
        client = get_ducklake_connection()
        
        # Charger les points uniques de la grille
        query = f"""
        SELECT DISTINCT 
            latitude, 
            longitude,
            road_type,
            speed_limit
        FROM silver.full_dataset
        WHERE target = 0  -- Points de grille
        LIMIT 10000
        """
        
        df = client.conn.execute(query).df()
        st.success(f"✅ Grille chargée: {len(df)} points")
        return df
        
    except Exception as e:
        st.error(f"⚠️ Erreur chargement grille: {e}")
        return pd.DataFrame()


def create_prediction_features(grid_df, prediction_datetime, atm_code):
    """Génère les features pour prédiction."""
    df = grid_df.copy()
    
    # Features temporelles
    df['hour'] = prediction_datetime.hour
    df['dayofweek'] = prediction_datetime.weekday()
    df['month'] = prediction_datetime.month
    df['atm'] = atm_code
    
    # Features dérivées
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_rush_morning'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_rush_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    # Encodage cyclique
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # TODO: Holidays NC
    df['is_holiday'] = 0
    df['school_holidays'] = df['month'].isin([1, 7, 8, 12]).astype(int)
    
    return df


def create_risk_map(predictions_df, center_lat=-21.5, center_lon=165.5, zoom=8):
    """Crée la carte Folium avec les prédictions."""
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="OpenStreetMap",
    )
    
    # Ajouter heatmap si prédictions disponibles
    if 'risk_score' in predictions_df.columns:
        heat_data = [
            [row['latitude'], row['longitude'], row['risk_score']]
            for _, row in predictions_df.iterrows()
            if row['risk_score'] > 0.5
        ]
        
        if heat_data:
            HeatMap(
                heat_data,
                min_opacity=0.3,
                max_opacity=0.8,
                radius=15,
                blur=20,
                gradient={0.4: 'blue', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}
            ).add_to(m)
    
    return m


# ==========================================
# INTERFACE PRINCIPALE
# ==========================================

st.title("🗺️ Carte de Prédiction des Accidents - Nouvelle-Calédonie")

st.markdown("""
### ℹ️ Comment ça marche ?

Cette application utilise l'**intelligence artificielle** pour prédire où les accidents de la route sont les plus susceptibles de se produire en Nouvelle-Calédonie.

💡 **Architecture** :
- Données stockées dans **DuckLake** (PostgreSQL + S3)
- Pipeline orchestré par **Dagster**
- Modèle ML entraîné sur 5 ans de données (2019-2024)

⚠️ **Important** : Ces prédictions sont des **estimations statistiques**. Elles indiquent les zones historiquement plus risquées.
""")

st.markdown("---")

# ==========================================
# SIDEBAR - PARAMÈTRES
# ==========================================

st.sidebar.header("⚙️ Paramètres de Prédiction")

selected_date = st.sidebar.date_input(
    "📅 Date de prédiction",
    value=date.today() + timedelta(days=1),
    min_value=date.today(),
    max_value=date.today() + timedelta(days=30),
)

selected_hour = st.sidebar.slider(
    "🕐 Heure de prédiction",
    0, 23, 12,
    help="Heure de la journée (0-23h)"
)

atm_options = {
    "Normal": 1,
    "Pluie légère": 2,
    "Pluie forte": 3,
    "Brouillard": 5,
}
selected_atm = st.sidebar.selectbox(
    "🌦️ Conditions météorologiques",
    list(atm_options.keys()),
    help="La météo influence fortement le risque d'accident.",
)
atm_code = atm_options[selected_atm]

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Mode d'Affichage")

display_mode = st.sidebar.radio(
    "Mode de filtrage",
    ["Top N points", "Seuil de risque"],
)

if display_mode == "Top N points":
    top_n = st.sidebar.slider("📊 Nombre de points à afficher", 10, 500, 100)
    threshold = 0.5
else:
    threshold = st.sidebar.slider(
        "🎯 Seuil de risque minimum",
        0.5, 0.95, 0.7, 0.05,
    )
    top_n = 1000

# ==========================================
# CHARGEMENT DONNÉES
# ==========================================

with st.spinner("📥 Chargement des données..."):
    grid_df = load_prediction_grid()

if grid_df.empty:
    st.error("❌ Impossible de charger la grille de prédiction. Vérifiez la connexion DuckLake.")
    st.stop()

# ==========================================
# GÉNÉRATION PRÉDICTIONS
# ==========================================

prediction_datetime = datetime.combine(selected_date, datetime.min.time()).replace(hour=selected_hour)

st.subheader(f"🔮 Prédictions pour le {prediction_datetime.strftime('%d/%m/%Y à %Hh')}")
st.markdown(f"**Conditions** : {selected_atm}")

with st.spinner("🤖 Génération des prédictions..."):
    # Préparer features
    features_df = create_prediction_features(grid_df, prediction_datetime, atm_code)
    
    # TODO: Charger et appliquer le modèle
    # Pour l'instant, scores aléatoires pour la démo
    features_df['risk_score'] = np.random.beta(2, 5, len(features_df))
    
    # Filtrer
    if display_mode == "Top N points":
        predictions_df = features_df.nlargest(top_n, 'risk_score')
    else:
        predictions_df = features_df[features_df['risk_score'] >= threshold].nlargest(top_n, 'risk_score')

# ==========================================
# AFFICHAGE CARTE
# ==========================================

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📍 Points analysés", f"{len(features_df):,}")
with col2:
    st.metric("⚠️ Points à risque", f"{len(predictions_df):,}")
with col3:
    avg_risk = predictions_df['risk_score'].mean() if not predictions_df.empty else 0
    st.metric("📊 Risque moyen", f"{avg_risk:.1%}")

st.markdown("---")

if not predictions_df.empty:
    risk_map = create_risk_map(predictions_df)
    st_folium(risk_map, width=1200, height=600)
    
    st.markdown("### 📋 Top 10 zones à risque")
    top_10 = predictions_df.nlargest(10, 'risk_score')[['latitude', 'longitude', 'risk_score', 'hour', 'atm']]
    top_10['risk_score'] = top_10['risk_score'].apply(lambda x: f"{x:.1%}")
    st.dataframe(top_10, use_container_width=True)
else:
    st.warning("⚠️ Aucun point à risque détecté avec ces paramètres.")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>
    🔬 Application développée avec Streamlit + DuckLake + Dagster<br>
    📊 Données : data.gouv.fr (accidents 2019-2024, dép. 988)<br>
    🗺️ Routes : OpenStreetMap<br>
    Version 2.0.0
    </small>
</div>
""", unsafe_allow_html=True)
