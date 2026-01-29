# Application Streamlit - Prédiction des Accidents NC

## 🎯 Description

Application web interactive pour visualiser les prédictions de zones à risque d'accidents en Nouvelle-Calédonie. Remplace le notebook Marimo `predict_map.py` avec une interface utilisateur moderne et intuitive.

## ✨ Fonctionnalités

### 🗺️ Carte Interactive
- Visualisation sur carte OpenStreetMap
- Couches horaires activables/désactivables
- Marqueurs colorés selon le niveau de risque :
  - 🔴 Rouge : Risque ÉLEVÉ (≥80%)
  - 🟠 Orange : Risque MOYEN (60-80%)
  - 🟡 Jaune : Risque FAIBLE (<60%)
- Popups détaillés avec informations de prédiction

### ⚙️ Paramètres Configurables
- **Date de prédiction** : Jusqu'à 30 jours dans le futur
- **Conditions météo** : Normal, Pluie légère, Pluie forte, Brouillard
- **Mode d'affichage** :
  - Top N par heure (recommandé) : Affiche les N points les plus dangereux
  - Seuil de probabilité : Affiche tous les points au-dessus d'un seuil

### 📊 Statistiques Détaillées
- Métriques globales (points à risque, risques moyen/max, heure critique)
- Statistiques par heure (nombre de points, risques min/moyen/max)
- Distribution des probabilités
- Top 10 des zones les plus dangereuses

## 🔧 Features ML Calculées

L'application calcule automatiquement **22 features enrichies** pour chaque prédiction :

### 1. Géographiques (2)
- `latitude`, `longitude`

### 2. Temporelles de base (3)
- `hour`, `dayofweek`, `month`

### 3. Météo (1)
- `atm` (conditions atmosphériques)

### 4. Interactions Spatio-Temporelles (12)
- `lat_hour`, `lon_hour` : Interactions position × heure
- `lat_dayofweek`, `lon_dayofweek` : Interactions position × jour
- `is_weekend`, `is_rush_morning`, `is_rush_evening`, `is_night` : Indicateurs temporels
- `hour_sin`, `hour_cos`, `dayofweek_sin`, `dayofweek_cos` : Encodage cyclique

### 5. Routes OSM (2)
- `road_type` : Type de route (motorway=5, trunk=4, primary=3, etc.)
- `speed_limit` : Limitation de vitesse

### 6. Densité et Proximité (2)
- `accident_density_5km` : Densité d'accidents dans un rayon de 5km
- `dist_to_noumea_km` : Distance à Nouméa

### 7. Temporelles Avancées (2)
- `is_holiday` : Jours fériés NC
- `school_holidays` : Vacances scolaires

## 📦 Installation

### Prérequis
- Python 3.13+
- Environnement virtuel activé

### Installation des dépendances
```bash
# Avec uv (recommandé)
uv add streamlit streamlit-folium

# Ou avec pip
pip install streamlit streamlit-folium
```

### Fichiers nécessaires
L'application nécessite ces fichiers à la racine du projet :
- `accident_model.pkl` : Modèle ML entraîné
- `atm_encoder.pkl` : Encodeur pour les conditions météo
- `features.pkl` : Liste des features du modèle
- `routes.nc` : Données OSM des routes (GeoJSON)

## 🚀 Lancement

```bash
streamlit run app_predict_map.py
```

L'application sera accessible sur `http://localhost:8501`

## 📖 Guide d'Utilisation

### 1️⃣ Configurer les Paramètres (Sidebar)
1. Sélectionner une **date de prédiction**
2. Choisir les **conditions météo** attendues
3. Sélectionner le **mode d'affichage** :
   - **Top N** : Affiche les 5 points les plus dangereux par heure (ajustable)
   - **Seuil** : Affiche tous les points > 70% de probabilité (ajustable)
4. Choisir les **heures à afficher** sur la carte

### 2️⃣ Analyser la Carte
- Cliquez sur le **panneau de couches** (en haut à droite) pour activer/désactiver les heures
- Cliquez sur les **marqueurs** pour voir les détails de prédiction
- Zoomez et déplacez la carte pour explorer les zones

### 3️⃣ Consulter les Statistiques
- **Métriques globales** : Vue d'ensemble rapide
- **Tableau par heure** : Statistiques détaillées heure par heure
- **Top 10** : Zones les plus dangereuses de la journée

## 🎨 Personnalisation

### Modifier la grille spatiale
Dans `load_model_and_data()` :
```python
grid_step = 0.02  # Résolution en degrés (~2km)
buffer_meters = 200  # Distance aux routes
```

### Ajuster les seuils de couleur
Dans `create_map()` :
```python
if row['probability'] >= 0.8:  # Risque ÉLEVÉ
    color = 'red'
elif row['probability'] >= 0.6:  # Risque MOYEN
    color = 'orange'
else:  # Risque FAIBLE
    color = 'yellow'
```

## ⚡ Optimisations

### Cache Streamlit
L'application utilise `@st.cache_resource` et `@st.cache_data` pour :
- Charger le modèle une seule fois
- Mettre en cache les prédictions pour chaque combinaison de paramètres

### Calcul Parallèle
Pour accélérer les prédictions sur 24 heures, utiliser :
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(calculate_features, ...) for hour in range(24)]
    results = [f.result() for f in futures]
```

## 🐛 Dépannage

### Erreur "ModuleNotFoundError: No module named 'streamlit_folium'"
```bash
uv add streamlit-folium
# ou
pip install streamlit-folium
```

### Carte vide ou pas de prédictions
- Vérifier que `routes.nc` existe et contient des données
- Diminuer le seuil de probabilité
- Augmenter le Top N

### Performances lentes
- Réduire `grid_step` (moins de points)
- Utiliser le mode "Top N" plutôt que "Seuil"
- Activer le calcul parallèle (voir section Optimisations)

## 📊 Comparaison avec predict_map.py

| Fonctionnalité | predict_map.py (Marimo) | app_predict_map.py (Streamlit) |
|---|---|---|
| Interface | Notebook réactif | Application web |
| Déploiement | Local uniquement | Local + Streamlit Cloud |
| Interactivité | Sliders natifs | UI moderne + sidebar |
| Carte | Folium intégré | streamlit-folium |
| Statistiques | Tableaux basiques | Métriques + graphiques |
| Cache | Manuel | Automatique (@st.cache) |
| Performance | Calcul à chaque cellule | Cache intelligent |

## 🚀 Déploiement en Production

### Streamlit Cloud (gratuit)
1. Pusher le code sur GitHub
2. Connecter sur [streamlit.io](https://streamlit.io)
3. Déployer depuis le repo GitHub

### Docker
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app_predict_map.py"]
```

### Configuration
Créer `.streamlit/config.toml` :
```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"

[server]
maxUploadSize = 200
enableCORS = false
```

## 📝 TODO / Améliorations Futures

- [ ] Charger la vraie densité d'accidents depuis `routes_with_features.pkl`
- [ ] Ajouter un slider de transparence pour les marqueurs
- [ ] Exporter les prédictions en CSV/GeoJSON
- [ ] Comparer plusieurs dates côte à côte
- [ ] Mode heatmap avec dégradé de couleur
- [ ] Notifications pour les heures critiques
- [ ] Intégration météo temps réel (API)
- [ ] Historique des prédictions passées

## 📄 Licence

Projet interne - Nouvelle-Calédonie

## 👥 Support

Pour toute question ou bug, contacter l'équipe ML.
