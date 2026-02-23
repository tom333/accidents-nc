# 🏛️ Architecture du Projet Accidents NC

Ce document détaille l'architecture technique du système de prédiction d'accidents routiers en Nouvelle-Calédonie.

---

## 📋 Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture médaillons (Bronze/Silver/Gold)](#architecture-médaillons)
3. [DuckLake (Lakehouse)](#ducklake-lakehouse)
4. [Pipeline Dagster](#pipeline-dagster)
5. [Infrastructure Kubernetes](#infrastructure-kubernetes)
6. [Applications](#applications)
7. [CI/CD GitOps](#cicd-gitops)
8. [Flux de données](#flux-de-données)

---

## 🎯 Vue d'ensemble

### Principes architecturaux

Le projet suit une **architecture moderne de data platform** basée sur :

1. **Architecture médaillons** : Bronze (raw) → Silver (features) → Gold (ML)
2. **Lakehouse pattern** : DuckLake (DuckDB + PostgreSQL + S3)
3. **Orchestration déclarative** : Dagster assets
4. **GitOps** : ArgoCD + Kubernetes
5. **CI/CD automatique** : GitHub Actions

### Stack technique complète

```
┌────────────────────────────────────────────────────────────────┐
│                         UTILISATEURS                            │
├────────────────────────────────────────────────────────────────┤
│  Data Scientists    │  Développeurs    │  Analystes Métier     │
└──────────┬──────────┴──────────┬───────┴───────────┬───────────┘
           │                     │                    │
           ↓                     ↓                    ↓
┌──────────────────────┬─────────────────┬──────────────────────┐
│  Dagster UI          │  Jupyter/Marimo │  Streamlit App       │
│  dagster.tgu.ovh     │  Notebooks      │  streamlit.tgu.ovh   │
│  (Orchestration)     │  (Exploration)  │  (Prédictions)       │
└──────────────────────┴─────────────────┴──────────────────────┘
           │                     │                    │
           └──────────┬──────────┴────────────────────┘
                      ↓
┌────────────────────────────────────────────────────────────────┐
│                   ORCHESTRATION LAYER                           │
├────────────────────────────────────────────────────────────────┤
│  Dagster (Kubernetes)                                           │
│  ├─ dagster-webserver   (UI + GraphQL API)                     │
│  ├─ dagster-daemon      (Schedules + Sensors)                  │
│  └─ dagster-user-code   (Assets + Resources)                   │
└────────────────────────────────────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────────────────────────┐
│                    COMPUTE LAYER                                │
├──────────────────────┬──────────────────┬──────────────────────┤
│  🥉 Bronze Assets    │  🥈 Silver       │  🥇 Gold Assets      │
│  Ingestion           │  Assets          │  ML Training         │
│  (Python/DuckDB)     │  Features Eng.   │  (CatBoost/Optuna)   │
└──────────────────────┴──────────────────┴──────────────────────┘
                      ↓
┌────────────────────────────────────────────────────────────────┐
│                     STORAGE LAYER                               │
├───────────────────────────────┬────────────────────────────────┤
│  DuckLake                     │  Metadata                      │
│  ├─ S3 (RustFS)              │  PostgreSQL Catalog            │
│  │   Parquet tables          │  ├─ Table schemas              │
│  │   Partitioning            │  ├─ Versions                   │
│  └─ DuckDB Engine            │  └─ Lineage                    │
└───────────────────────────────┴────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────────────────────────┐
│                   INFRASTRUCTURE LAYER                          │
├────────────────────────────────────────────────────────────────┤
│  Kubernetes (microk8s)                                          │
│  ├─ Namespace: dagster       (Dagster services)                │
│  ├─ Namespace: datalab       (PostgreSQL)                      │
│  ├─ Namespace: ia-lab        (RustFS S3)                       │
│  └─ ArgoCD                   (GitOps CD)                       │
└────────────────────────────────────────────────────────────────┘
```

---

## 🥉🥈🥇 Architecture Médaillons

### Concept

L'architecture médaillons (Medallion Architecture) organise les données en **3 couches de qualité croissante** :

```
External Data → 🥉 Bronze → 🥈 Silver → 🥇 Gold → ML Models
     (CSV)       (Raw)     (Features)  (Datasets)  (Production)
```

### 🥉 Bronze Layer (Raw Data)

**Rôle** : Ingestion et stockage des données brutes sans transformation

**Source** : 
- CSV data.gouv.fr (2019-2024)
- 2 fichiers par année : `caracteristiques-YYYY.csv` + `usagers-YYYY.csv`

**Transformations minimales** :
- Parsing dates françaises (`jour/mois/an hrmn` → datetime)
- Nettoyage coordonnées (`,` → `.` pour lat/lon)
- Filtrage département 988 (Nouvelle-Calédonie)

**Tables créées** :
- `bronze.caracteristiques` : Métadonnées accidents
- `bronze.usagers` : Victimes par accident
- `bronze.accidents_nc` : Join caracteristiques + usagers (dep=988)

**Code** : [src/accidents/bronze/ingest.py](../src/accidents/bronze/ingest.py)

**Exemple de données bronze** :
```python
# bronze.accidents_nc
{
    'Num_Acc': 2024000123456,
    'date_accident': datetime(2024, 6, 15, 7, 30),
    'lat': -22.2758,
    'long': 166.4580,
    'atm': 1,  # Normal
    'dep': '988'
}
```

### 🥈 Silver Layer (Curated Data)

**Rôle** : Enrichissement avec features spatiales et temporelles

**Transformations** :

1. **Features temporelles** (12 colonnes)
   ```python
   hour_of_day          # 0-23
   day_of_week          # 0-6 (0=lundi)
   month                # 1-12
   is_weekend           # bool
   is_rush_hour         # 7-9h ou 17-19h
   hour_sin, hour_cos   # Encodage cyclique
   day_sin, day_cos
   month_sin, month_cos
   ```

2. **Features OSM** (réseau routier)
   - Téléchargement 30+ communes NC via OSMnx
   - Buffer 200m autour routes
   - Join spatial avec grille 0.02° (~2.2km)
   - Features : `road_length`, `road_count`, `nearest_road_type`

3. **Échantillonnage négatif intelligent**
   - Exclusion spatiale : 300m autour accidents réels
   - Distribution temporelle : 85% heures à risque / 15% aléatoire
   - Ratio : 2.2 négatifs pour 1 positif
   - Target : 0 (pas accident) vs 1 (accident)

**Tables créées** :
- `silver.features` : Dataset complet avec features + négatifs

**Code** : 
- [src/accidents/silver/features.py](../src/accidents/silver/features.py)
- [src/accidents/silver/negatives.py](../src/accidents/silver/negatives.py)

**Exemple de données silver** :
```python
# silver.features
{
    'latitude': -22.2758,
    'longitude': 166.4580,
    'hour_of_day': 7,
    'day_of_week': 5,  # Samedi
    'month': 6,
    'is_weekend': True,
    'is_rush_hour': True,
    'hour_sin': 0.707,
    'hour_cos': 0.707,
    'atm': 1,
    'road_length': 1250.5,
    'road_count': 3,
    'nearest_road_type': 'primary',
    'target': 1  # Accident
}
```

### 🥇 Gold Layer (Business Data)

**Rôle** : Datasets prêts pour le ML + modèles entraînés

**Transformations** :

1. **Préparation datasets**
   - Encodage `atm` (One-Hot ou Label Encoding)
   - Split train/test (80/20) stratifié
   - Suppression colonnes identifiantes (Num_Acc)
   - Sauvegarde encoders (`.pkl`)

2. **Training modèles**
   - Optimisation hyperparamètres (Optuna - 50 trials)
   - Algorithmes : CatBoost, LightGBM, XGBoost
   - Métrique : Recall (priorité détection accidents)
   - Logging MLflow
   - Export modèle final (`.pkl`)

**Tables créées** :
- `gold.train` : Dataset entraînement
- `gold.test` : Dataset test
- `gold.feature_metadata` : Métadonnées features

**Artefacts** :
- `accident_model.pkl` : Modèle CatBoost final
- `atm_encoder.pkl` : Encoder conditions météo
- `features.pkl` : Liste features utilisées

**Code** : 
- [src/accidents/gold/datasets.py](../src/accidents/gold/datasets.py)
- [src/accidents/gold/training.py](../src/accidents/gold/training.py)

**Exemple de données gold** :
```python
# gold.train
X_train: DataFrame (2400 rows × 24 features)
y_train: Series (2400 rows) - binary target

# gold.test
X_test: DataFrame (600 rows × 24 features)
y_test: Series (600 rows) - binary target
```

---

## 🦆 DuckLake (Lakehouse)

### Qu'est-ce que DuckLake ?

**DuckLake = DuckDB + PostgreSQL (catalog) + S3 (storage)**

C'est une implémentation moderne de **lakehouse** qui combine :
- **Data lake** : Stockage S3 peu coûteux
- **Data warehouse** : Performance DuckDB
- **Catalog** : Métadonnées PostgreSQL

### Architecture DuckLake

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT APPLICATIONS                       │
│  Dagster Assets │ Jupyter Notebooks │ Streamlit App         │
└────────────┬────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────┐
│                   DuckDB ENGINE                              │
│  ├─ SQL Query Engine                                        │
│  ├─ Parquet Reader/Writer                                   │
│  ├─ S3 Integration (httpfs extension)                       │
│  └─ PostgreSQL Scanner (attach database)                    │
└────────────┬───────────────────────────┬────────────────────┘
             │                           │
             ↓                           ↓
┌────────────────────────────┐  ┌──────────────────────────┐
│  S3 STORAGE (RustFS)       │  │  POSTGRESQL CATALOG      │
│  rustfs.tgu.ovh:443        │  │  postgres.datalab:5432   │
│  ├─ accidents/bronze/      │  │  ├─ ducklake.tables      │
│  ├─ accidents/silver/      │  │  ├─ ducklake.schemas     │
│  └─ accidents/gold/        │  │  └─ ducklake.versions    │
│  Format: Parquet           │  │  Metadata only           │
└────────────────────────────┘  └──────────────────────────┘
```

### Fonctionnement

#### 1. Écriture de données

```python
from src.accidents.ducklake import get_client

client = get_client()

# Écrire dans DuckLake
client.write(
    df,                          # Polars DataFrame
    table="bronze.accidents_nc", # Nom de table
    mode="replace"               # replace | append
)
```

**Ce qui se passe en interne** :

1. DuckDB écrit `df` en Parquet vers S3 : `s3://accidents/bronze/accidents_nc/data.parquet`
2. PostgreSQL catalog enregistre :
   ```sql
   INSERT INTO ducklake.tables VALUES (
       'bronze', 
       'accidents_nc', 
       's3://accidents/bronze/accidents_nc/',
       'parquet',
       1234567,  -- row_count
       NOW()
   );
   ```

#### 2. Lecture de données

```python
# Lire depuis DuckLake
df = client.table("bronze.accidents_nc").pl()
```

**Ce qui se passe en interne** :

1. PostgreSQL catalog retourne le path S3
2. DuckDB lit directement depuis S3 :
   ```sql
   SELECT * FROM read_parquet('s3://accidents/bronze/accidents_nc/*.parquet');
   ```

### Avantages DuckLake

| Fonctionnalité | Avantage |
|----------------|----------|
| **Stockage S3** | Coût faible, scalabilité illimitée |
| **Format Parquet** | Compression ~10x, lecture columnar rapide |
| **DuckDB** | Performance imbattable (SQL analytique) |
| **Catalog PostgreSQL** | Métadonnées versionnées, lineage |
| **Schema evolution** | Ajout colonnes sans récriture |
| **Partitioning** | Pruning automatique (ex: par date) |

### Configuration

[src/accidents/config.py](../src/accidents/config.py) :

```python
from src.accidents.config import get_config

config = get_config()
config.postgres_url
```

---

## 🎭 Pipeline Dagster

### Qu'est-ce que Dagster ?

**Dagster** est un orchestrateur de data pipelines moderne basé sur le concept d'**assets** (et non de tasks comme Airflow).

**Philosophie** : "Données d'abord, compute ensuite"

### Assets vs Tasks

**Airflow (tasks)** :
```python
extract_task >> transform_task >> load_task
# Focus sur les tâches, pas les données
```

**Dagster (assets)** :
```python
@asset
def bronze_data(): ...

@asset
def silver_data(bronze_data): ...
# Focus sur les données (assets), Dagster dérive le DAG
```

### Assets du Projet

```python
# dagster_pipeline/assets/bronze.py
@asset
def bronze_accidents_nc() -> int:
    """Asset Bronze : Ingestion CSV + filtrage dep 988"""
    from src.accidents.bronze.ingest import ingest_all
    return ingest_all()

# dagster_pipeline/assets/silver.py
@asset
def silver_features(bronze_accidents_nc: int) -> int:
    """Asset Silver : Features temporelles + OSM + négatifs"""
    from src.accidents.silver.features import build_feature_store
    return build_feature_store()

# dagster_pipeline/assets/gold.py
@asset
def gold_train_test(silver_features: int) -> dict:
    """Asset Gold : Split train/test stratifié"""
    from src.accidents.gold.datasets import create_ml_datasets
    return create_ml_datasets()

@asset
def gold_models(gold_train_test: dict) -> str:
    """Asset Gold : Training CatBoost + Optuna"""
    from src.accidents.gold.training import train_best_model
    return train_best_model(gold_train_test)
```

### DAG Généré Automatiquement

```
bronze_accidents_nc
       ↓
  silver_features
       ↓
  gold_train_test
       ↓
   gold_models
```

Dagster **dérive automatiquement le DAG** à partir des dépendances entre assets (paramètres de fonctions).

### Resources Dagster

[dagster_pipeline/resources/ducklake.py](../dagster_pipeline/resources/ducklake.py) :

```python
from dagster import ConfigurableResource
from src.accidents.ducklake import DuckLakeClient

class DuckLakeResource(ConfigurableResource):
    """Resource Dagster pour DuckLake"""
    
    def get_client(self) -> DuckLakeClient:
        return get_client()

# Utilisation dans asset
@asset
def my_asset(ducklake: DuckLakeResource):
    client = ducklake.get_client()
    df = client.table("bronze.accidents_nc").pl()
    ...
```

### Schedules

[dagster_pipeline/schedules.py](../dagster_pipeline/schedules.py) :

```python
from dagster import ScheduleDefinition, DefaultScheduleStatus

# Entraînement hebdomadaire (dimanche 3h)
weekly_training_schedule = ScheduleDefinition(
    job=train_models_job,
    cron_schedule="0 3 * * 0",  # Dimanche 3h
    default_status=DefaultScheduleStatus.RUNNING,
)

# Prédictions quotidiennes (6h)
daily_predictions_schedule = ScheduleDefinition(
    job=predict_daily_job,
    cron_schedule="0 6 * * *",
    default_status=DefaultScheduleStatus.RUNNING,
)
```

---

## ☸️ Infrastructure Kubernetes

### Cluster microk8s

**Pourquoi microk8s ?**
- Kubernetes léger (single-node)
- Idéal pour serveur dédié
- Extensions intégrées (DNS, storage, ingress)

**Namespaces** :

```
dagster/         # Dagster services
├── dagster-webserver-*
├── dagster-daemon-*
└── dagster-user-deployment-*

datalab/         # PostgreSQL (DuckLake catalog)
└── postgresql-*

ia-lab/          # RustFS (S3)
└── rustfs-*
```

### Dagster sur Kubernetes

#### Architecture

```
┌────────────────────────────────────────────────────────┐
│                    INGRESS                              │
│  dagster.tgu.ovh → dagster-webserver:3000              │
└────────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────┐
│  dagster-webserver (Deployment)                        │
│  ├─ UI React                                           │
│  ├─ GraphQL API                                        │
│  └─ PostgreSQL storage backend                        │
└────────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────┐
│  dagster-daemon (Deployment)                           │
│  ├─ Schedule Runner (cron jobs)                        │
│  ├─ Sensor Runner (event triggers)                     │
│  └─ Run Coordinator                                    │
└────────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────┐
│  dagster-user-deployment (Deployment)                  │
│  ├─ User code (assets + resources)                     │
│  ├─ Python 3.13 + dependencies                         │
│  └─ K8s Run Launcher (spawn pods)                      │
└────────────────────────────────────────────────────────┘
```

#### Manifests K8s

[infra/k8s/dagster/deployment-dagster-user.yaml](../infra/k8s/dagster/deployment-dagster-user.yaml) :

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dagster-user-deployment
  namespace: dagster
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: dagster-user-code
        image: localhost:32000/accidents-dagster:latest
        env:
        - name: POSTGRES_URL
          value: postgresql://accidents:***@postgres.datalab.svc.cluster.local:5432/accidents
        - name: S3_ENDPOINT
          value: https://rustfs.tgu.ovh
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: ducklake-secrets
              key: AWS_ACCESS_KEY_ID
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: ducklake-secrets
              key: AWS_SECRET_ACCESS_KEY
        volumeMounts:
        - name: dagster-models
          mountPath: /opt/dagster/models
      volumes:
      - name: dagster-models
        persistentVolumeClaim:
          claimName: dagster-models-pvc
```

#### ConfigMap DuckLake

[infra/k8s/dagster/configmap-dagster-ducklake.yaml](../infra/k8s/dagster/configmap-dagster-ducklake.yaml) :

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: dagster-ducklake-config
  namespace: dagster
data:
  POSTGRES_URL: postgresql://accidents:***@postgres.datalab.svc.cluster.local:5432/accidents
  S3_ENDPOINT: https://rustfs.tgu.ovh
  S3_BUCKET: accidents
```

---

## 🌐 Applications

### Streamlit App

**URL** : https://streamlit.tgu.ovh

**Code** : [apps/streamlit/app.py](../apps/streamlit/app.py)

**Fonctionnalités** :
- Sélection date et météo
- Prédictions par heure (0-23h)
- Carte Folium interactive
- Marqueurs colorés par risque
- Tableau récapitulatif

**Architecture** :
```
Streamlit App
    ↓
DuckLake (lecture gold.models)
    ↓
Modèle CatBoost (accident_model.pkl)
    ↓
Prédictions (probabilités)
    ↓
Carte Folium (visualisation)
```

**Deployment K8s** :

[infra/k8s/streamlit/deployment.yaml](../infra/k8s/streamlit/deployment.yaml) :

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: streamlit-app
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: streamlit
        image: localhost:32000/accidents-streamlit:latest
        ports:
        - containerPort: 8501
        env:
        - name: POSTGRES_URL
          valueFrom:
            configMapKeyRef:
              name: streamlit-config
              key: POSTGRES_URL
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: streamlit-secrets
              key: AWS_ACCESS_KEY_ID
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: streamlit-secrets
              key: AWS_SECRET_ACCESS_KEY
```

### API FastAPI (à venir)

**Endpoint** : `/predict`

**Payload** :
```json
{
  "latitude": -22.2758,
  "longitude": 166.4580,
  "datetime": "2024-06-15T07:30:00",
  "atm": 1
}
```

**Response** :
```json
{
  "probability": 0.85,
  "risk_level": "high",
  "features": {...}
}
```

---

## 🚀 CI/CD GitOps

### Workflow Complet

```
┌─────────────────────────────────────────────────────────────┐
│  1. DÉVELOPPEUR                                              │
│  git commit -m "feat: nouvelle feature"                     │
│  git push origin main                                        │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  2. GITHUB ACTIONS - CI TESTS (.github/workflows/)          │
│  ├─ Lint (ruff)                                             │
│  ├─ Tests unitaires (pytest + coverage)                     │
│  └─ Tests validation (phases 1-6)                           │
│  Durée: ~5 minutes                                          │
└──────────────────────────┬──────────────────────────────────┘
                           ↓ (si success)
┌─────────────────────────────────────────────────────────────┐
│  3. GITHUB ACTIONS - CI BUILD                               │
│  ├─ Build Dockerfile.dagster                                │
│  ├─ Build Dockerfile.streamlit                              │
│  ├─ Scan sécurité Trivy                                     │
│  ├─ Tag images (sha, branch, latest)                        │
│  └─ Push vers registry localhost:32000                      │
│  Durée: ~10 minutes                                         │
└──────────────────────────┬──────────────────────────────────┘
                           ↓ (si success)
┌─────────────────────────────────────────────────────────────┐
│  4. GITHUB ACTIONS - CD UPDATE MANIFESTS                    │
│  ├─ sed image tags dans K8s YAML                            │
│  ├─ git commit "chore: update image tags"                   │
│  └─ git push origin main                                    │
│  Durée: ~1 minute                                           │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  5. ARGOCD - AUTO-SYNC (3min polling)                       │
│  ├─ Détecte changement manifests Git                        │
│  ├─ Compare état Git vs Cluster K8s                         │
│  ├─ Applique différences (kubectl apply)                    │
│  └─ Redémarre pods Dagster/Streamlit                        │
│  Durée: ~5 minutes                                          │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  6. KUBERNETES - DÉPLOIEMENT                                │
│  ├─ Téléchargement nouvelles images                         │
│  ├─ Rolling update (zero downtime)                          │
│  └─ Health checks                                           │
│  Durée: ~2 minutes                                          │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
                    ✅ DEPLOYED !
                    (~22 minutes total)
```

### ArgoCD Application

[infra/k8s/dagster/argocd-application.yaml](../infra/k8s/dagster/argocd-application.yaml) :

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: dagster-accidents-pipeline
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/tom333/accidents-nc.git
    targetRevision: main
    path: infra/k8s/dagster
  destination:
    server: https://kubernetes.default.svc
    namespace: dagster
  syncPolicy:
    automated:
      prune: true       # Supprime ressources obsolètes
      selfHeal: true    # Répare divergences
    syncOptions:
    - CreateNamespace=true
```

**Auto-sync activé** : ArgoCD déploie automatiquement dès que Git change (polling 3 min).

---

## 📊 Flux de Données

### Vue d'ensemble end-to-end

```
┌─────────────────────────────────────────────────────────────┐
│  DATA SOURCES                                                │
├─────────────────────────────────────────────────────────────┤
│  data.gouv.fr                                                │
│  ├─ caracteristiques-2019.csv                               │
│  ├─ caracteristiques-2020.csv                               │
│  ├─ ...                                                      │
│  ├─ usagers-2019.csv                                         │
│  └─ usagers-2020.csv                                         │
│                                                              │
│  OpenStreetMap (OSMnx API)                                   │
│  └─ Réseau routier 30+ communes NC                          │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  🥉 BRONZE LAYER                                            │
├─────────────────────────────────────────────────────────────┤
│  Asset: bronze_accidents_nc                                  │
│  ├─ Ingestion CSV DuckDB (read_csv + union_by_name)        │
│  ├─ Parsing dates françaises (strptime)                     │
│  ├─ Nettoyage coordonnées (trim, replace comma)            │
│  └─ Filtrage dep='988'                                      │
│                                                              │
│  Output: bronze.accidents_nc (~1000 rows)                   │
│  Storage: S3 Parquet s3://accidents/bronze/                 │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  🥈 SILVER LAYER                                            │
├─────────────────────────────────────────────────────────────┤
│  Asset: silver_features                                      │
│  ├─ Features temporelles (12 colonnes)                      │
│  │   hour, day, month, weekend, rush_hour, sin/cos         │
│  ├─ Features OSM (téléchargement réseau routier)           │
│  │   road_length, road_count, nearest_road_type            │
│  ├─ Échantillonnage négatif intelligent                     │
│  │   Exclusion 300m, ratio temporel 85/15, ratio 2.2:1     │
│  └─ Join spatial (buffer 200m)                              │
│                                                              │
│  Output: silver.features (~3200 rows = 1000 + 2200)        │
│  Storage: S3 Parquet s3://accidents/silver/                 │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  🥇 GOLD LAYER - DATASETS                                   │
├─────────────────────────────────────────────────────────────┤
│  Asset: gold_train_test                                      │
│  ├─ Encodage atm (One-Hot)                                  │
│  ├─ Drop NA + colonnes identifiantes                        │
│  ├─ Split stratifié 80/20 (train/test)                      │
│  └─ Sauvegarde encoders (.pkl)                              │
│                                                              │
│  Output:                                                     │
│  ├─ gold.train (~2400 rows × 24 features)                  │
│  ├─ gold.test (~600 rows × 24 features)                    │
│  └─ atm_encoder.pkl, features.pkl                           │
│  Storage: S3 Parquet s3://accidents/gold/                   │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  🥇 GOLD LAYER - TRAINING                                   │
├─────────────────────────────────────────────────────────────┤
│  Asset: gold_models                                          │
│  ├─ Optuna hyperparameter tuning (50 trials)                │
│  │   CatBoost, LightGBM, XGBoost                            │
│  ├─ Métrique: Recall (priorité détection accidents)         │
│  ├─ Training best model (CatBoost)                          │
│  ├─ Evaluation (accuracy 98.2%, recall 87%)                 │
│  └─ Logging MLflow                                           │
│                                                              │
│  Output:                                                     │
│  ├─ accident_model.pkl (CatBoost final)                     │
│  ├─ MLflow artifacts (metrics, params, plots)               │
│  └─ gold_models table (metadata)                            │
│  Storage: S3 + K8s PVC /opt/dagster/models/                 │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  APPLICATIONS                                                │
├─────────────────────────────────────────────────────────────┤
│  Streamlit App                                               │
│  ├─ Load accident_model.pkl                                 │
│  ├─ Génère grille prédictions 24h                           │
│  ├─ Top N points à risque par heure                         │
│  └─ Carte Folium interactive                                │
│                                                              │
│  API FastAPI (à venir)                                       │
│  ├─ Endpoint /predict                                        │
│  └─ Prédiction temps réel                                   │
└─────────────────────────────────────────────────────────────┘
```

### Métriques de Pipeline

| Étape | Durée | Rows In | Rows Out | Storage |
|-------|-------|---------|----------|---------|
| Bronze Ingestion | ~30s | 5000 CSV | 1000 | 2 MB Parquet |
| Silver Features | ~5min | 1000 | 3200 | 8 MB Parquet |
| Gold Datasets | ~10s | 3200 | 3200 | 10 MB Parquet |
| Gold Training | ~2min | 2400 | 1 model | 50 MB .pkl |

**Total pipeline** : ~8 minutes (ingestion → modèle final)

---

## � Flux de Données

### Vue d'ensemble end-to-end

*(Section déjà documentée ci-dessus dans "Flux de données")*

---

## �🔐 Sécurité

### Secrets Management

**Kubernetes Secrets** :
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ducklake-secrets
  namespace: dagster
type: Opaque
stringData:
  AWS_ACCESS_KEY_ID: ***
  AWS_SECRET_ACCESS_KEY: ***
```

**Injection dans pods** :
```yaml
env:
- name: AWS_ACCESS_KEY_ID
  valueFrom:
    secretKeyRef:
      name: ducklake-secrets
      key: AWS_ACCESS_KEY_ID
- name: AWS_SECRET_ACCESS_KEY
  valueFrom:
    secretKeyRef:
      name: ducklake-secrets
      key: AWS_SECRET_ACCESS_KEY
```

### Scan Sécurité

**Trivy** dans CI Build :
```yaml
- name: Run Trivy vulnerability scanner
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: localhost:32000/accidents-dagster:${{ github.sha }}
    format: 'sarif'
    output: 'trivy-results.sarif'

- name: Upload Trivy results to GitHub Security
  uses: github/codeql-action/upload-sarif@v2
  with:
    sarif_file: 'trivy-results.sarif'
```

---

## 📈 Monitoring (à venir)

### Métriques Dagster

- **Asset materialization success rate**
- **Pipeline execution time**
- **Data quality checks**

### Métriques Modèle

- **Prediction latency**
- **Model accuracy drift**
- **Feature importance evolution**

### Infrastructure

- **CPU/Memory usage pods**
- **S3 storage growth**
- **PostgreSQL query performance**

---

## 🎓 Références

- [Dagster Documentation](https://docs.dagster.io/)
- [DuckDB Documentation](https://duckdb.org/docs/)
- [Medallion Architecture (Databricks)](https://www.databricks.com/glossary/medallion-architecture)
- [ArgoCD Documentation](https://argo-cd.readthedocs.io/)
- [Lakehouse Architecture (Databricks)](https://www.databricks.com/blog/2020/01/30/what-is-a-data-lakehouse.html)
