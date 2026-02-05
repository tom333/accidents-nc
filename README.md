# 🚦 Prédiction d'Accidents Routiers - Nouvelle-Calédonie

Système de prédiction des zones à risque d'accidents routiers en Nouvelle-Calédonie (département 988) utilisant l'apprentissage automatique, les données OSM et les statistiques gouvernementales françaises.

## 📊 Vue d'Ensemble

Ce projet implémente un **classificateur binaire géospatial-temporel** pour prédire le risque d'accident par localisation et heure. Il combine :

- **Données officielles** : 5 ans d'accidents de data.gouv.fr (2019-2024)
- **Réseau routier OSM** : 30+ communes de Nouvelle-Calédonie
- **Features temporelles** : Heure, jour de la semaine, mois, conditions météo
- **Échantillonnage négatif intelligent** : Exclusion spatiale (300m) et distribution temporelle (85/15)
- **Orchestration Dagster** : Pipeline industrialisé sur Kubernetes avec architecture médaillons
- **DuckLake** : Lakehouse moderne (DuckDB + PostgreSQL catalog + S3 storage)

### 🎯 Performances du Modèle

**Meilleur modèle** : CatBoost (optimisé avec Optuna - 50 trials)

| Métrique | Accident | Pas Accident | Global |
|----------|----------|--------------|--------|
| **Recall** | **87.0%** | 99.7% | - |
| **Precision** | 97.9% | 98.3% | - |
| **F1-Score** | 0.92 | 0.99 | - |
| **Accuracy** | - | - | **98.2%** |

**Résultat clé** : Détecte **272/312 accidents réels** (87%), avec seulement 40 faux négatifs et 6 faux positifs.

**Optimisation** : Hyperparamètres optimisés automatiquement via Optuna (50 essais, 74s d'entraînement).

---

## 🏗️ Architecture du Projet

### 🐳 Infrastructure (Production)

```
Kubernetes Cluster (microk8s)
├── Namespace: dagster
│   ├── dagster-webserver       # UI Dagster (https://dagster.tgu.ovh)
│   ├── dagster-daemon          # Scheduler & sensors
│   └── dagster-user-deployment # User code (assets bronze/silver/gold)
├── Namespace: datalab
│   └── postgresql              # DuckLake catalog (métadonnées tables)
└── Namespace: ia-lab
    └── rustfs-svc              # S3 compatible (https://rustfs.tgu.ovh)
```

### 📂 Code Source

```
accidents/
├── dagster_accidents/          # Assets Dagster (orchestration)
│   ├── assets_prod.py          # Bronze/Silver/Gold (sans training)
│   ├── assets.py               # Version complète avec gold_models
│   └── repository.py           # Definitions Dagster
├── pipeline/                   # Logique métier (stage_ingest, stage_features, etc.)
│   ├── config.py               # Configuration DuckLake
│   ├── stage_ingest.py         # 🥉 Bronze: ingestion CSV
│   ├── stage_features.py       # 🥈 Silver: enrichissement + négatifs
│   ├── stage_datasets.py       # 🥇 Gold: train/test splits
│   └── stage_modeling.py       # 🥇 Gold: entraînement modèles
├── k8s/dagster/                # Manifests Kubernetes
│   ├── dagster-helm-app-updated.yaml  # ArgoCD Application
│   ├── configmap-dagster-ducklake.yaml
│   └── DEPLOY_DAGSTER.md
├── Dockerfile.dagster          # Image user-code (1.12GB)
├── rebuild-and-deploy.sh       # Script rebuild/redeploy automatique
├── accident_fetch_data.py      # Notebook Marimo legacy (exploration)
├── predict_map.py              # Visualisation interactive (Marimo)
├── routes.nc                   # Cache OSM (~50MB, git-ignoré)
├── INDUSTRIALISATION.md        # Guide Dagster + K8s
└── README.md                   # Cette documentation
```

### 🏛️ Architecture Médaillons (Lakehouse)

**DuckLake = DuckDB + PostgreSQL (catalog) + S3 (storage)**

```mermaid
graph LR
    A[CSV data.gouv.fr] -->|ingest_all| B[🥉 Bronze<br/>bronze.accidents_nc]
    B -->|build_feature_store| C[🥈 Silver<br/>silver.features]
    C -->|build_datasets| D[🥇 Gold<br/>gold.train/test]
    D -->|run_training| E[🥇 Gold<br/>gold_models + MLflow]
    
    B -.-> F[(PostgreSQL<br/>Catalog)]
    C -.-> F
    D -.-> F
    E -.-> F
    
    B -.-> G[(S3 RustFS<br/>Parquet)]
    C -.-> G
    D -.-> G
```

- **🥉 Bronze** : Données brutes (5 ans CSV, dep=988, datetime parsing)
- **🥈 Silver** : Features enrichies (OSM buffer 200m, négatifs 22k ratio, temporelles)
- **🥇 Gold** : Datasets ML (80/20 split, encodage atm) + Modèles (CatBoost/LGBM/XGB)

### 🔄 Workflow de Développement

```bash
# 1. Modifier le code dans pipeline/ ou dagster_accidents/
vim pipeline/stage_ingest.py

# 2. Rebuild & redeploy automatique
./rebuild-and-deploy.sh

# 3. Matérialiser via UI Dagster
# → https://dagster.tgu.ovh > Sélectionner asset > "Materialize"
```

---

## 🚀 Installation

### Prérequis

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (gestionnaire de paquets)
- Docker + Kubernetes (microk8s recommandé pour local)
- Accès à un cluster PostgreSQL (DuckLake catalog)
- Accès à un stockage S3 compatible (DuckLake data storage)

### Installation des dépendances

```bash
# Cloner le projet
git clone <repo-url>
cd accidents

# Installer les dépendances (production uniquement)
uv sync

# Ou avec dépendances training (optuna, catboost, etc.)
uv sync --extra training

# Ou avec dev tools (marimo, dagster-webserver, pytest)
uv sync --extra dev

# Activer l'environnement virtuel
source .venv/bin/activate
```

### Dépendances principales

**Core (production)** :
```toml
dagster>=1.8.0       # Orchestration
dagster-k8s          # Run launcher Kubernetes
dagster-postgres     # Storage backend
duckdb>=1.3.0        # Moteur SQL + DuckLake
geopandas, osmnx     # Analyse géospatiale
scikit-learn         # Machine learning
polars, pandas       # Manipulation de données
pyarrow              # DuckDB ↔ Polars
boto3                # Client S3
```

**Training (optionnel)** :
```toml
catboost, lightgbm, xgboost  # Gradient boosting
optuna                        # Hyperparameter tuning
torch, pytorch-tabnet         # Deep learning
imblearn                      # Resampling
```

**Dev (optionnel)** :
```toml
marimo[recommended]   # Notebooks réactifs (exploration)
dagster-webserver     # UI locale
pytest                # Tests
```

---

## 📚 Utilisation

### 🔁 Nouveau pipeline modulaire (DuckDB)

Les 4 notebooks Marimo du dossier `notebooks/` fonctionnent comme une chaîne : chacun consomme les tables DuckDB créées par le précédent avant d'écrire ses propres outputs. Toutes les données transitent par `data/accidents_pipeline.duckdb`.

| Étape | Notebook | Tables écrites | Description |
|-------|----------|----------------|-------------|
| 1 | `01_ingest_raw.py` | `raw.caracteristiques`, `raw.usagers`, `raw.accidents_nc` | Ingestion CSV/URLs et normalisation des colonnes clés (datetime, lat/lon, atm). |
| 2 | `02_enrich_features.py` | `features.full_dataset` | Génération des négatifs spatialement sûrs, rattachement des features OSM/densité, ajout des features temporelles avancées. |
| 3 | `03_prepare_datasets.py` | `datasets.train`, `datasets.test`, `datasets.feature_metadata` | Encodage `atm`, drop NA, split stratifié, sauvegarde `atm_encoder.pkl` et `features.pkl`. |
| 4 | `04_model_training.py` | Logs MLflow + artefacts `.pkl` | Tuning Optuna (CatBoost/LGBM/XGBoost) + ensembles TabNet/MLP/logistic, export `accident_model.pkl`. |

Pour exécuter l’ensemble :

```bash
marimo run notebooks/01_ingest_raw.py
marimo run notebooks/02_enrich_features.py
marimo run notebooks/03_prepare_datasets.py
marimo run notebooks/04_model_training.py
```

Chaque notebook peut aussi être ouvert en mode interactif (`marimo edit ...`) pour ajuster les paramètres et visualiser les outputs intermédiaires (comptes DuckDB, previews, etc.).

### 1️⃣ Entraînement du Modèle

> 💡 **Recommandé** : exécuter les 4 notebooks modulaires (`01_` → `04_`) pour bénéficier du stockage DuckDB et de la reprise par étape.
>
> Le notebook ci-dessous reste disponible pour des itérations rapides mais sera progressivement retiré.

**Notebook Marimo (legacy)** : `accident_fetch_data.py`

```bash
marimo edit accident_fetch_data.py
```

**Pipeline complet** (14 cellules) :

1. **Ingestion** : Fusion 6 CSV data.gouv.fr via DuckDB
2. **Nettoyage** : Parsing dates françaises, filtrage coordonnées
3. **Réseau OSM** : Téléchargement 30+ communes, buffer 200m
4. **Grille spatiale** : Résolution 0.02° (~2.2km), spatial join
5. **Échantillonnage négatif** :
   - Exclusion spatiale 300m autour accidents
   - Distribution temporelle 85% heures à risque / 15% aléatoire
   - Ratio 2:1 (négatifs/positifs)
6. **Features engineering** : 
   - **6 features de base** : latitude, longitude, hour, dayofweek, month, atm
   - **18 features enrichies** :
     - Interactions spatio-temporelles (lat×hour, lon×dayofweek, etc.)
     - Attributs OSM (type de route, vitesse limite)
     - Métriques de densité et proximité
     - Encodage cyclique temporel (sin/cos)
     - Indicateurs temporels avancés (jours fériés, vacances)
   - **Total : 24 features**
7. **Optimisation hyperparamètres** :
   - Framework **Optuna** avec MedianPruner
   - 50 essais par algorithme (CatBoost, LightGBM, XGBoost)
   - Métrique : **Recall** (priorité détection accidents)
   - Sélection automatique du meilleur modèle
8. **Évaluation** : Rapport classification, courbe ROC, importance features
9. **Export** : Modèles sauvegardés en `.pkl`

**Configuration optimisée** :

```python
CONFIG = {
    'n_negative_samples_ratio': 22000,   # Ratio réaliste basé sur taux d'accidents réel
    'buffer_meters': 200,
    'grid_step': 0.02,
    'accident_exclusion_buffer_km': 0.3, # 300m au lieu de 500m
    'temporal_risk_ratio': 0.85          # 85% heures à risque
}
```

### 🗺️ Visualisation Interactive

**Notebook Marimo** : `predict_map.py`

```bash
marimo edit predict_map.py
```

**Interface utilisateur** :

- 📅 **Sélecteur de date** : N'importe quelle date
- 🎯 **Mode de sélection** :
  - **Top N** (recommandé) : Affiche les N points les plus dangereux par heure (1-10, défaut=3)
  - **Seuil** : Probabilité minimale (50-95%, défaut=70%)
- 🌦️ **Conditions météo** : Normal, Pluie légère, Pluie forte, Brouillard

**Carte Folium interactive** :

- Marqueurs colorés par risque (rouge ≥80%, orange 60-80%, jaune <60%)
- Filtres par heure (panneau de contrôle)
- Popups avec détails (heure, probabilité)
- Statistiques dynamiques (nombre de points, probabilités min/max/moyenne)

**Outputs** :

- Carte interactive avec 24 couches (1 par heure)
- Tableau récapitulatif par heure
- Résumé global (heure la plus dangereuse, risques moyen/max)

### 🤖 Prédictions Automatiques (À faire)

**Script Python** : `predict_daily.py` (nécessite adaptation pour DuckLake)

```bash
# Prédictions pour demain (conditions normales)
python predict_daily.py

# Date spécifique
python predict_daily.py --date 2026-01-25

# Avec conditions météo
python predict_daily.py --atm 2  # 1=Normal, 2=Pluie légère, 3=Pluie forte, 5=Brouillard

# Base DuckDB personnalisée
python predict_daily.py --db custom.duckdb
```

**Pipeline automatique** :

1. Charge modèle + grille routière (~1500 points)
2. Génère **24h × 1500 = 36,000 prédictions**
3. Stocke dans DuckDB avec index optimisés
4. Affiche statistiques (risque moyen, points critiques)

**Automatisation cron** (23h chaque jour) :

```bash
crontab -e
# Ajouter :
0 23 * * * cd /path/to/accidents && python predict_daily.py >> predict.log 2>&1
```

### 🔍 Consultation des Données DuckLake

**DuckDB CLI** (voir [QUERIES.md](QUERIES.md) pour plus d'exemples) :

```bash
# Se connecter au catalog DuckLake (depuis un pod k8s)
kubectl exec -it -n dagster deployment/dagster-user-deployment-accidents -- \
  duckdb -c "ATTACH 'postgres://user:password@postgresql.datalab:5432/data' AS ducklake (TYPE postgres); SELECT * FROM ducklake.bronze.accidents_nc LIMIT 10;"

# Ou depuis un notebook Marimo local
marimo edit accident_fetch_data.py
# Dans le notebook:
import duckdb
from pipeline.config import ensure_connection
conn = ensure_connection()
conn.execute("SELECT * FROM bronze.accidents_nc LIMIT 10").pl()
```

**Requêtes utiles** :

```sql
-- Compter les accidents par année
SELECT EXTRACT(YEAR FROM event_time) as year, COUNT(*) as nb_accidents
FROM bronze.accidents_nc
GROUP BY year
ORDER BY year;

-- Top 10 features les plus importantes
SELECT * FROM silver.features WHERE target = 1 LIMIT 10;

-- Distribution train/test
SELECT 'train' as split, COUNT(*) as rows FROM gold.train
UNION ALL
SELECT 'test' as split, COUNT(*) as rows FROM gold.test;
```

**Stockage S3** :

```bash
# Lister les fichiers Parquet sur RustFS
aws --endpoint-url=https://rustfs.tgu.ovh s3 ls s3://accidents-bucket/ducklake/ --recursive

# Télécharger une table localement
aws --endpoint-url=https://rustfs.tgu.ovh s3 cp \
  s3://accidents-bucket/ducklake/bronze/accidents_nc/ . --recursive
```

---

## 📐 Schéma de Données

### DuckLake (PostgreSQL Catalog + S3 Parquet Storage)

**Catalog PostgreSQL** : `postgresql.datalab.svc.cluster.local:5432/data`  
**Storage S3** : `s3://accidents-bucket/ducklake/` (RustFS)

#### 🥉 Schema Bronze

```sql
-- Table principale: accidents New Caledonia
CREATE TABLE bronze.accidents_nc (
    Num_Acc VARCHAR,              -- ID accident (clé)
    event_time TIMESTAMP,         -- Datetime parsé (format français)
    latitude DOUBLE,              -- Coordonnée WGS84
    longitude DOUBLE,             -- Coordonnée WGS84
    atm INTEGER,                  -- Conditions météo (1-9)
    year INTEGER,                 -- 2019-2024
    PRIMARY KEY (Num_Acc)
);

-- Tables sources (CSV bruts)
CREATE TABLE bronze.caracteristiques (...);  -- Métadonnées accident
CREATE TABLE bronze.usagers (...);           -- Victimes
```

#### 🥈 Schema Silver

```sql
CREATE TABLE silver.features (
    -- Identifiant
    row_id INTEGER PRIMARY KEY,
    
    -- Target
    target INTEGER,               -- 0=pas accident, 1=accident
    
    -- Features géographiques
    latitude DOUBLE,
    longitude DOUBLE,
    dist_to_noumea_km DOUBLE,
    accident_density_5km DOUBLE,
    
    -- Features temporelles
    hour INTEGER,                 -- 0-23
    dayofweek INTEGER,            -- 0-6
    month INTEGER,                -- 1-12
    atm INTEGER,                  -- 1-9
    hour_sin DOUBLE,              -- Encodage cyclique
    hour_cos DOUBLE,
    is_weekend BOOLEAN,
    is_rush_hour BOOLEAN,
    
    -- Features OSM
    road_type VARCHAR,
    speed_limit INTEGER,
    
    -- Features d'interaction
    lat_hour DOUBLE,
    lon_dayofweek DOUBLE,
    hour_dayofweek DOUBLE,
    -- ... (18 features d'interaction total)
);
```

#### 🥇 Schema Gold

```sql
CREATE TABLE gold.train (
    -- Mêmes colonnes que silver.features
    -- 80% des données (split stratifié)
);

CREATE TABLE gold.test (
    -- Mêmes colonnes que silver.features
    -- 20% des données (split stratifié)
);

CREATE TABLE gold.feature_metadata (
    feature_name VARCHAR PRIMARY KEY,
    feature_order INTEGER,
    feature_type VARCHAR          -- 'numeric', 'categorical', 'interaction'
);
```

**Artefacts stockés en S3** :
- `s3://accidents-bucket/ducklake/artifacts/atm_encoder.pkl`
- `s3://accidents-bucket/ducklake/artifacts/features.pkl`
- `s3://accidents-bucket/ducklake/artifacts/accident_model.pkl` (si training)

### Features ML

**24 features enrichies** (6 de base + 18 calculées) :

```python
# Features géographiques de base (2)
features_base = ['latitude', 'longitude']

# Features temporelles de base (4)
features_temporelles = ['hour', 'dayofweek', 'month', 'atm']

# Interactions spatio-temporelles (12)
features_interactions = [
    'lat_hour', 'lon_hour',           # Géographie × heure
    'lat_dayofweek', 'lon_dayofweek', # Géographie × jour
    'lat_month', 'lon_month',         # Géographie × mois
    'hour_dayofweek', 'hour_month',   # Heure × jour/mois
    'dayofweek_month',                # Jour × mois
    'lat_lon', 'hour_dayofweek_month', 'lat_lon_hour'
]

# Encodage cyclique (4)
features_cycliques = [
    'hour_sin', 'hour_cos',           # Continuité 23h→0h
    'dayofweek_sin', 'dayofweek_cos'  # Continuité dimanche→lundi
]

# Attributs OSM (2)
features_osm = ['road_type', 'speed_limit']

# Métriques spatiales (2)
features_spatiales = [
    'accident_density_5km',    # Densité historique
    'nearest_accident_km'      # Distance accident le plus proche
]

# Indicateurs temporels avancés (6)
features_temporelles_avancees = [
    'is_weekend',              # Samedi/Dimanche
    'is_rush_morning',         # 7h-9h
    'is_rush_evening',         # 17h-19h
    'is_night',                # 22h-6h
    'is_holiday',              # Jours fériés NC
    'school_holidays'          # Vacances scolaires
]

# Distance aux centres urbains (1)
features_distance = ['dist_to_noumea_km']
```

---

## 🔧 Détails Techniques

### Stratégie d'Échantillonnage Négatif

**Problème** : Classifier route "normale" vs "accident" nécessite des contre-exemples réalistes.

**Solution** :

1. **Exclusion spatiale** :
   - Buffer 300m autour de chaque accident
   - Grille filtrée : ne garder que points ≥300m de tout accident historique
   - Évite faux négatifs (zones réellement dangereuses)

2. **Distribution temporelle hybride** :
   - 85% : Timestamps échantillonnés depuis accidents réels (heures à risque)
   - 15% : Timestamps uniformes sur période complète (couvrir heures sûres)
   - Équilibre entre ciblage et diversité

3. **Ratio réaliste** :
   - 22000:1 (reflète le taux d'accidents réel)
   - Calibration des probabilités pour prédictions réalistes
   - Évite la sur-prédiction (était 360× trop avec ratio 2:1)

### Importance des Features (Top 10 / 24 features)

| Feature | Importance | Interprétation |
|---------|------------|----------------|
| `latitude` | 32.1% | Position géographique principale |
| `longitude` | 24.8% | Zones urbaines vs rurales |
| `lat_lon` | 12.3% | Interaction géographique |
| `dayofweek` | 8.7% | Week-end vs semaine |
| `road_type` | 6.2% | Type de route (OSM) |
| `hour_dayofweek` | 4.5% | Interaction temporelle |
| `dist_to_noumea_km` | 3.8% | Proximité centre urbain |
| `speed_limit` | 2.1% | Vitesse autorisée |
| `accident_density_5km` | 1.9% | Historique local |
| `hour_sin` | 1.6% | Cyclicité horaire |

**Insights** :
- **Géographie** : 69.2% (lat/lon/interactions) → forte concentration spatiale
- **Features OSM** : 8.3% (road_type + speed_limit) → gain significatif
- **Interactions** : 18.4% → synergie spatio-temporelle capturée
- **14 features restantes** : 4.1% → contribution marginale mais utile pour cas limites

### Comparaison d'Algorithmes (Optuna 50 trials)

| Modèle | Recall Accident | Precision | F1-Score | AUC | Temps Total (s) |
|--------|----------------|-----------|----------|-----|------------------|
| **CatBoost** ⭐ | **87.0%** | 97.9% | 0.92 | 0.974 | 74 |
| LightGBM | 86.2% | 97.6% | 0.91 | 0.972 | 58 |
| XGBoost | 84.8% | 97.1% | 0.90 | 0.968 | 92 |

**Configuration Optuna** :
- 50 essais par algorithme
- MedianPruner (arrêt précoce si performance < médiane)
- Optimisation métrique : Recall (classe minoritaire)
- Espace de recherche : learning_rate, max_depth, n_estimators, etc.

**Sélection automatique** : CatBoost sélectionné (meilleur recall en 74s).

---

## 📊 Résultats Détaillés

### Matrice de Confusion

```
                Prédit Non    Prédit Oui
Réel Non         2282 ✅        6 ⚠️
Réel Oui          40 ❌       272 ✅
```

- **True Positives** : 272 accidents détectés (87.0%)
- **False Negatives** : 40 accidents ratés (12.8%)
- **False Positives** : 6 fausses alarmes (0.26%)
- **True Negatives** : 2282 non-accidents corrects (99.7%)

### Courbe ROC

- **AUC-ROC** : 0.973
- Excellent compromis sensibilité/spécificité

### Cas d'Usage

**Campagnes de prévention** : Cibler les 10% de zones/heures les plus à risque pour allouer 70% des ressources.

**Signalisation dynamique** : Afficher alertes en temps réel sur zones à risque élevé (≥80%).

**Étude d'impact** : Évaluer l'effet de nouvelles infrastructures sur le risque prédit.

---

## 🗺️ Données Sources

### Accidents (data.gouv.fr)

- **Caractéristiques** : 6 CSV annuels (2019-2024)
- **Usagers** : Détails victimes
- **Filtrage** : `dep='988'` (Nouvelle-Calédonie)
- **Format dates** : `jour/mois/an hrmn` (français)

### Réseau Routier (OSM)

- **30+ communes** : Nouméa, Dumbéa, Mont-Dore, Païta, etc.
- **Type** : `network_type='drive'`
- **Cache** : `routes.nc` (GeoJSON, ~50MB)
- **Fallback** : Province Sud/Nord si commune échoue

### Grille Spatiale

- **Résolution** : 0.02° (~2.2km)
- **Étendue** : lat ∈ [-23.0, -19.5], lon ∈ [163.5, 168.0]
- **Buffer routes** : 200m (EPSG:3857)
- **Points finaux** : ~1500 sur routes

---

## � Déploiement Kubernetes

### Configuration Initiale

1. **Créer le Secret avec credentials** :

```bash
kubectl create secret generic rustfs-credentials-dagster -n dagster \
  --from-literal=AWS_ACCESS_KEY_ID=<your-key> \
  --from-literal=AWS_SECRET_ACCESS_KEY=<your-secret> \
  --from-literal=POSTGRES_PASSWORD=<pg-password>
```

2. **Créer le ConfigMap** :

```bash
kubectl apply -f k8s/dagster/configmap-dagster-ducklake.yaml
```

3. **Déployer via ArgoCD** :

```bash
# Appliquer l'Application ArgoCD
kubectl apply -f k8s/dagster/dagster-helm-app-updated.yaml

# Vérifier le sync
argocd app get dagster -n argocd
```

### Rebuild & Redeploy (Développement)

**Script automatisé** : `rebuild-and-deploy.sh`

```bash
# Éditer le code
vim pipeline/stage_ingest.py

# Build + push + restart
./rebuild-and-deploy.sh

# Résultat:
# ✅ Build Docker: ~45s
# ✅ Push registry: ~20s  
# ✅ Rollout restart: ~60s
# ✅ Total: ~2min
```

**Vérification** :

```bash
# Status des pods
kubectl get pods -n dagster | grep user-deployment

# Logs en temps réel
kubectl logs -f -n dagster deployment/dagster-user-deployment-accidents

# Tester connectivité DuckLake
kubectl exec -n dagster deployment/dagster-user-deployment-accidents -- \
  python -c "from pipeline.config import ensure_connection; conn = ensure_connection(); print('✅ OK')"
```

### Troubleshooting

**Erreur : "No module named 'X'"**
→ Ajouter la dépendance dans `pyproject.toml` puis rebuild

**Erreur : "fe_sendauth: no password supplied"**
→ Vérifier que `POSTGRES_PASSWORD` est dans le Secret

**Erreur : "Could not parse string '...' according to format"**
→ Vérifier le parsing datetime dans `pipeline/stage_ingest.py`

**Pod en CrashLoopBackOff**
→ `kubectl logs -n dagster <pod-name>` pour voir l'erreur exacte

### Documentation Complète

Voir [INDUSTRIALISATION.md](INDUSTRIALISATION.md) pour :
- Architecture détaillée Dagster + DuckLake
- Configuration ArgoCD
- Manifests Kubernetes
- Stratégies de déploiement production

## �🛠️ Développement

### Structure du Code

**Séparation des responsabilités** :

```
pipeline/           → Logique métier (réutilisable)
├── config.py       → Configuration DuckLake
├── stage_*.py      → Étapes du pipeline (ingest/features/datasets/modeling)
└── utils.py        → Fonctions utilitaires

dagster_accidents/  → Orchestration
├── assets_prod.py  → Assets production (bronze/silver/gold)
├── assets.py       → Assets complets (avec training)
└── repository.py   → Definitions Dagster
```

**Principes** :
- `pipeline/` = source de vérité (indépendant de Dagster)
- `dagster_accidents/` = wrappers minces (juste `@asset` + appels)
- Assets retournent des `dict` (métriques pour logs Dagster)

### Ajout d'un Nouvel Asset

1. **Créer la fonction métier** dans `pipeline/stage_<name>.py` :

```python
# pipeline/stage_predictions.py
from .config import ensure_connection

def generate_predictions():
    conn = ensure_connection()
    conn.execute("""
        CREATE OR REPLACE TABLE gold.predictions AS
        SELECT * FROM silver.features WHERE ...
    """)
    return {'predictions': 1500}
```

2. **Wrapper Dagster** dans `dagster_accidents/assets_prod.py` :

```python
from pipeline.stage_predictions import generate_predictions

@asset(
    name="gold_predictions",
    key_prefix=["gold"],
    deps=[gold_datasets],
    group_name="gold"
)
def gold_predictions(context: AssetExecutionContext) -> dict:
    stats = generate_predictions()
    context.log.info(f"[GOLD] Predictions: {stats}")
    return stats
```

3. **Rebuild & redeploy** :

```bash
./rebuild-and-deploy.sh
```

### Tests Locaux (sans K8s)

```bash
# Configurer DuckLake en local (fichier .env)
cat > .env << EOF
DUCKLAKE_DATABASE_URL=duckdb:///data/accidents_pipeline.duckdb
DUCKLAKE_DATA_PATH=./data/ducklake
EOF

# Tester une fonction pipeline directement
python -c "from pipeline.stage_ingest import ingest_all; print(ingest_all())"

# Ou lancer Dagster en local
uv run dagster dev -m dagster_accidents.repository -p 3000
```

### Structure Marimo (Notebooks Exploration)

Les notebooks Marimo utilisent une syntaxe réactive :

```python
@app.cell
def _(dependencies):
    # Code ici
    return variables_exportées
```

**Règles** :
- Variables uniques dans tout le notebook
- Dernière expression = output affiché
- `mo.ui.*` pour éléments interactifs

**Usage recommandé** : Exploration uniquement, pas pour la production.

---

## 📝 Améliorations Futures

### Infrastructure & DevOps

- [x] Dagster sur Kubernetes ✅
- [x] Architecture médaillons (Bronze/Silver/Gold) ✅
- [x] DuckLake (PostgreSQL catalog + S3 storage) ✅
- [x] Script rebuild-and-deploy automatique ✅
- [ ] CI/CD avec GitLab CI / GitHub Actions
- [ ] Tests unitaires pipeline (pytest)
- [ ] Monitoring Prometheus + Grafana
- [ ] Alerting sur échecs de matérialisation

### Features Géospatiales Avancées

- [x] Type de route (OSM : primary, secondary, residential) ✅
- [x] Vitesse limite ✅
- [x] Distance au centre urbain (Nouméa) ✅
- [x] Densité d'accidents historiques (rayon 5km) ✅
- [ ] Présence d'intersections (rayon 100m)
- [ ] Courbure de la route
- [ ] Pente/dénivelé

**Gain observé** : +0.5% recall (87.0% vs 86.5%)

### Features Temporelles

- [x] Jours fériés (Nouvelle-Calédonie) ✅
- [x] Vacances scolaires ✅
- [x] Heures de pointe (matin/soir) ✅
- [x] Encodage cyclique (continuité temporelle) ✅
- [ ] Événements spéciaux (festivals, matchs)
- [ ] Conditions météo historiques (température, précipitations)

### Modèles Alternatifs

- [x] Optuna hyperparameter tuning (CatBoost, LightGBM, XGBoost) ✅
- [ ] Stacking/Blending (ensemble des 3 meilleurs)
- [ ] TabNet (deep learning pour tabular)
- [ ] Modèles géospatiaux (GWR, spatial lag)
- [ ] AutoML (AutoGluon, H2O)
- [ ] MLflow pour versioning et registry des modèles

### Applications

- [ ] API REST (FastAPI) avec prédictions temps réel
- [ ] Dashboard Streamlit avec cartes interactives
- [ ] Notifications automatiques zones critiques
- [ ] Intégration avec systèmes de signalisation dynamique
- [ ] Scheduler Dagster pour prédictions quotidiennes automatiques

---

## 📄 Licence

Ce projet utilise des données publiques sous licence Open Data (data.gouv.fr) et OpenStreetMap (ODbL).

---

## 🙏 Remerciements

- **data.gouv.fr** : Données officielles accidents
- **OpenStreetMap** : Réseau routier
- **Dagster** : Orchestration moderne des pipelines data
- **DuckDB** : Moteur SQL performant avec support DuckLake
- **Marimo** : Framework notebooks réactifs pour exploration

---

## 📧 Contact

Pour questions, suggestions ou contributions, ouvrez une issue sur le dépôt GitHub.

**Dernière mise à jour** : Février 2026 - Version Dagster/Kubernetes industrialisée