# 🚦 Accidents NC - Prédiction d'Accidents Routiers

**Système de prédiction des zones à risque d'accidents routiers en Nouvelle-Calédonie** utilisant l'apprentissage automatique, les données OSM et les statistiques gouvernementales françaises.

[![CI Tests](https://github.com/tom333/accidents-nc/workflows/CI%20Tests/badge.svg)](https://github.com/tom333/accidents-nc/actions)
[![CI Build](https://github.com/tom333/accidents-nc/workflows/CI%20Build/badge.svg)](https://github.com/tom333/accidents-nc/actions)

---

## 📊 Vue d'Ensemble

### Concept

Ce projet implémente un **classificateur binaire géospatial-temporel** pour prédire le risque d'accident par localisation et heure. Il combine :

- **5 ans de données officielles** (2019-2024) depuis data.gouv.fr (département 988)
- **Réseau routier OSM** de 30+ communes de Nouvelle-Calédonie
- **Features temporelles avancées** (heure, jour, météo, encodage cyclique)
- **Échantillonnage négatif intelligent** (exclusion spatiale 300m, ratio temporel 85/15)
- **Architecture médaillons** (Bronze → Silver → Gold) avec DuckLake
- **Orchestration Dagster** sur Kubernetes avec CI/CD GitOps

### 🎯 Performances du Modèle

**Approche** : **Ensemble Blending** (3 modèles combinés avec pondération optimale)

**Modèles de base** :
- **CatBoost** (poids: 0.4) - Optimisé avec Optuna (50 trials)
- **XGBoost** (poids: 0.4) - Gradient boosting classique
- **MLP** (poids: 0.2) - Réseau de neurones avec embeddings catégoriels

**Formule blending** :
```python
P_final = (0.4 × P_catboost + 0.4 × P_xgboost + 0.2 × P_mlp) / 1.0
```

**Performances de l'ensemble** :

| Métrique | Valeur |
|----------|--------|
| **Recall** | **92.2%** |
| **Precision** | 94.5% |
| **F1-Score** | **93.3%** |
| **AUC-ROC** | **98.0%** |
| **Seuil optimal** | 0.641 |

**AUC-ROC par modèle** :

| Modèle | AUC-ROC |
|--------|---------|
| CatBoost | 97.8% |
| XGBoost | 97.7% |
| MLP | 97.3% |
| **Blend (ensemble)** | **98.0%** |

**Résultat clé** : Le blend combine la force de chaque modèle et dépasse tous les modèles individuels en AUC-ROC (+0.2pp vs CatBoost).

### 🧐 Analyse des Résultats

> **TL;DR : Ces résultats sont excellents** pour un problème de prédiction spatiale sur données déséquilibrées.

**Ce que chaque métrique signifie concrètement :**

| Métrique | Interprétation terrain |
|----------|------------------------|
| **Recall 92.2%** | Sur 100 zones dangereuses réelles, le modèle en détecte **92**. Seulement **8 sont manquées** (faux négatifs). |
| **Precision 94.5%** | Sur 100 alertes émises, **94–95 correspondent à une vraie zone à risque**. Très peu de fausses alarmes. |
| **AUC-ROC 98.0%** | Le modèle discrimine quasi-parfaitement les zones à risque sur l’ensemble des seuils. Un score ≥ 95% est considéré excellent ; ≥ 98% est de niveau production. |
| **Seuil 0.641** | Optimisé (vs 0.5 par défaut), il réduit les fausses alarmes tout en maintenant un recall élevé. |

**Pourquoi le blend apporte quelque chose ?** Les 3 modèles font des erreurs sur des exemples *différents*. En les combinant (pondération 40/40/20), le blend "vote" de façon plus robuste : une zone difficile a bien moins de chances d’être mal classifiée par les **trois** modèles simultanément.

**Ce qui reste imparfait :** 8% de faux négatifs (zones à risque non détectées) et 5.5% de fausses alarmes.
Dans un contexte de sécurité routière, manquer une zone à risque est plus coûteux qu’une fausse alerte — le recall de **92.2%** est donc la métrique à maximiser en priorité.

---


## 🏗️ Architecture

### Stack Technique

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATIONS                              │
├──────────────────────┬──────────────────────────────────────┤
│  Dagster UI          │  Streamlit App                        │
│  dagster.tgu.ovh     │  streamlit.tgu.ovh                   │
└──────────────────────┴──────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              ORCHESTRATION (Dagster)                         │
├──────────────────────┬──────────────────────────────────────┤
│  dagster-webserver   │  dagster-daemon                       │
│  dagster-user-code   │  (schedules, sensors)                │
└──────────────────────┴──────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              DATA PIPELINE (Médaillons)                      │
├──────────────────────┬──────────────┬───────────────────────┤
│  🥉 Bronze           │  🥈 Silver   │  🥇 Gold              │
│  src/assets/         │  src/assets/ │  src/assets/          │
│  bronze/             │  silver/     │  gold/                │
│  ├─ ingest.py        │  ├─ features │  ├─ datasets.py       │
│  └─ raw data         │  └─ enrich   │  └─ training.py       │
└──────────────────────┴──────────────┴───────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 STORAGE (DuckLake)                           │
├──────────────────────┬──────────────────────────────────────┤
│  S3 (RustFS)         │  PostgreSQL                           │
│  rustfs.tgu.ovh      │  Catalog métadonnées                 │
│  Tables Parquet      │  Tables/versions                     │
└──────────────────────┴──────────────────────────────────────┘
```

**Technologies clés** :
- **Dagster** : Orchestration data pipeline
- **DuckLake** : Lakehouse moderne (DuckDB + PostgreSQL catalog + S3 storage)
- **Kubernetes** : Déploiement production (microk8s)
- **ArgoCD** : GitOps (CI/CD automatique)
- **Streamlit** : Interface utilisateur prédictions
- **OSMnx** : Téléchargement réseau routier
- **CatBoost / XGBoost / MLP** : Ensemble blending (3 modèles optimisés)
- **Optuna** : Hyperparameter tuning automatique

### Architecture Médaillons

```
CSV data.gouv.fr → 🥉 Bronze → 🥈 Silver → 🥇 Gold → Modèles ML → Blending
                    (raw)     (features)  (datasets)  (CB/XGB/MLP)  (Ensemble)
```

**Détails** : Voir [docs/architecture.md](docs/architecture.md)

---

## 📂 Structure du Projet

```
accidents/
├── src/                        # 🎯 Code source consolidé
│   ├── assets/                 # Assets Dagster (Médaillons)
│   │   ├── bronze/             # Ingestion (DuckDB)
│   │   ├── silver/             # Feature Engineering
│   │   └── gold/               # Datasets ML + Models
│   ├── resources/              # Ressources (DuckLake)
│   ├── utils/                  # Utilitaires (Spatiaux, etc.)
│   ├── definitions.py          # Point d'entrée Dagster
│   ├── schedules.py            # Jobs schedulés
│   ├── ducklake.py             # Client DuckLake common
│   └── config.py               # Configuration
│
├── apps/                       # 🌐 Applications
│   ├── api/                    # FastAPI (prédictions)
│   │   └── main.py
│   └── streamlit/              # Interface utilisateur
│       └── app.py
│
├── exploration/                # 🔬 Notebooks Marimo
│   ├── 01_eda_accidents.py
│   ├── 02_features_engineering.py
│   ├── 03_model_experiments.py
│   └── 04_viz_predictions.py
│
├── infra/                      # 🐳 Infrastructure
│   ├── docker/
│   │   ├── Dockerfile.dagster
│   │   ├── Dockerfile.streamlit
│   │   └── docker-compose.dev.yml
│   └── k8s/
│       ├── dagster/            # Manifests Dagster
│       └── streamlit/          # Manifests Streamlit
│
├── tests/                      # 🧪 Tests
│   ├── unit/
│   │   ├── test_bronze.py
│   │   ├── test_silver.py
│   │   └── test_gold.py
│   ├── integration/
│   └── test_phase*_validation.py
│
├── docs/                       # 📚 Documentation
│   ├── architecture.md         # Architecture détaillée
│   ├── workflow.md             # Workflow développement
│   └── deployment.md           # Guide déploiement
│
├── .github/workflows/          # 🤖 CI/CD
│   ├── ci-tests.yml            # Tests automatiques
│   ├── ci-build.yml            # Build Docker images
│   └── cd-update-manifests.yml # GitOps ArgoCD
│
├── pyproject.toml              # Configuration projet
└── README.md                   # Cette documentation
```

---

## 🚀 Quickstart

### Prérequis

- **Python 3.13+**
- **[uv](https://github.com/astral-sh/uv)** (gestionnaire de paquets)
- **Docker** (pour développement local)
- **Kubernetes** (pour déploiement, optionnel)

### Installation

```bash
# 1. Cloner le projet
git clone https://github.com/tom333/accidents-nc.git
cd accidents

# 2. Installer les dépendances
uv sync

# 3. Activer l'environnement
source .venv/bin/activate
```

### Développement Local

#### Option A : Infrastructure Docker Compose (recommandé)

Démarre PostgreSQL + MinIO S3 en local pour simuler DuckLake :

```bash
cd infra/docker
docker-compose -f docker-compose.dev.yml up -d

# Vérifier les services
docker-compose ps
# → postgres (5432), minio (9000/9001)

# Lancer Dagster UI
cd ../..
PYTHONPATH=$PWD dagster dev -f src/definitions.py

# Ouvrir http://localhost:3000
```

#### Option B : Notebooks Marimo (exploration)

```bash
# EDA et features engineering
marimo edit exploration/01_eda_accidents.py

# Entraînement modèle
marimo edit exploration/03_model_experiments.py

# Visualisation prédictions
marimo edit exploration/04_viz_predictions.py
```

### Lancer les Tests

```bash
# Tests unitaires
uv run pytest tests/unit/ -v

# Avec coverage
uv run pytest tests/unit/ --cov=src/accidents --cov-report=html

# Voir coverage
open htmlcov/index.html

# Lint code
uv run ruff check src/ tests/
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [docs/architecture.md](docs/architecture.md) | Architecture détaillée (médaillons, DuckLake, Kubernetes) |
| [docs/workflow.md](docs/workflow.md) | Workflow de développement (Git, Dagster, Tilt) |
| [docs/deployment.md](docs/deployment.md) | Guide déploiement Kubernetes + ArgoCD |
| [TILT.md](TILT.md) | Développement rapide avec Tilt (live-reload) |
| [DOCKER.md](DOCKER.md) | Build et push images Docker |


---

## 🔄 Workflow CI/CD

### Pipeline Automatique

```
Push main → CI Tests → CI Build → CD Update Manifests → ArgoCD Deploy
   ↓           ↓           ↓              ↓                  ↓
 Commit     Pytest      Docker         sed K8s           Auto-sync
            Ruff        Trivy         Git push           3min
```

**Détails** :
1. **CI Tests** ([.github/workflows/ci-tests.yml](.github/workflows/ci-tests.yml))
   - Lint avec ruff
   - Tests unitaires avec pytest + coverage
   - Tests validation phases 1-6

2. **CI Build** ([.github/workflows/ci-build.yml](.github/workflows/ci-build.yml))
   - Build Dockerfile.dagster et Dockerfile.streamlit
   - Scan sécurité Trivy
   - Tags automatiques (sha, branch, semver)

3. **CD Update Manifests** ([.github/workflows/cd-update-manifests.yml](.github/workflows/cd-update-manifests.yml))
   - Met à jour image tags dans K8s manifests
   - Commit + push vers main
   - ArgoCD détecte et déploie automatiquement

**Timeline** : ~22 minutes du push au déploiement complet

---

## 🎯 Utilisation

### Matérialiser le Pipeline Dagster

1. **Accéder à l'UI Dagster** : https://dagster.tgu.ovh (ou http://localhost:3000 en local)

2. **Matérialiser les assets** :
   - Sélectionner `bronze_accidents_nc` → Cliquer "Materialize"
   - Sélectionner `silver_features` → Cliquer "Materialize"
   - Sélectionner `ml_datasets` → Cliquer "Materialize"
   - Sélectionner `tune_catboost` → Cliquer "Materialize"
   - Sélectionner `tune_xgboost` → Cliquer "Materialize" (optionnel)
   - Sélectionner `tune_mlp` → Cliquer "Materialize" (optionnel)
   - Sélectionner `blend_model` → Cliquer "Materialize" (pour l'ensemble complet)

3. **Vérifier les données dans DuckLake** :
   ```python
   from src.ducklake import get_client

   client = get_client()
   df = client.table("bronze.accidents_nc").limit(10).pl()
   print(df)
   ```

### Lancer l'Application Streamlit

```bash
# Local
cd apps/streamlit
streamlit run app.py

# Ouvrir http://localhost:8501
```

**Production** : https://streamlit.tgu.ovh

**Fonctionnalités** :
- 📅 Sélecteur de date
- 🌦️ Conditions météo (Normal, Pluie, Brouillard)
- 🎯 Mode Top N ou Seuil probabilité
- 🗺️ Carte Folium interactive avec marqueurs de risque
- 📊 Tableau récapitulatif par heure
- 🔮 Prédictions par ensemble blending (CatBoost/XGBoost/MLP)

---

## 🔧 Configuration

### Variables d'Environnement

Créer `.env` à la racine du projet :

```bash
# DuckLake PostgreSQL catalog
POSTGRES_URL=postgresql://accidents:***@postgres.datalab.svc.cluster.local:5432/accidents

# S3 Storage (RustFS)
S3_ENDPOINT=https://rustfs.tgu.ovh
S3_BUCKET=accidents
AWS_ACCESS_KEY_ID=***
AWS_SECRET_ACCESS_KEY=***

# Dagster
DAGSTER_POSTGRES_USER=dagster
DAGSTER_POSTGRES_PASSWORD=***
DAGSTER_POSTGRES_DB=dagster
DAGSTER_HOME=/opt/dagster/dagster_home
```

### Configuration DuckLake

[src/config.py](src/config.py) :

```python
from src.config import get_config

config = get_config()
config.postgres_url
```

---

## 🧪 Tests

### Structure

```
tests/
├── unit/                       # Tests unitaires (mocks)
│   ├── test_bronze.py          # Ingestion
│   ├── test_silver.py          # Features
│   └── test_gold.py            # Datasets + Training
├── integration/                # Tests intégration (DuckLake réel)
└── test_phase*_validation.py   # Tests validation phases
```

### Commandes

```bash
# Tous les tests unitaires
uv run pytest tests/unit/ -v

# Test spécifique
uv run pytest tests/unit/test_bronze.py::TestIngestCaracteristiques -v

# Tests intégration (marqués @pytest.mark.integration)
uv run pytest tests/integration/ -v

# Tests avec coverage
uv run pytest --cov=src/accidents --cov-report=term-missing

# Tests validation phases
python tests/test_phase1_validation.py
python tests/test_phase6_validation.py
```

---

## 🐳 Développement avec Tilt (live-reload)

**Workflow ultra-rapide** : 2-5 secondes au lieu de 2 minutes pour rebuild complet.

```bash
# Installer Tilt
curl -fsSL https://raw.githubusercontent.com/tilt-dev/tilt/master/scripts/install.sh | bash

# Démarrer
tilt up

# Interface web : http://localhost:10350
# Les modifications de code Python sont synchro automatiquement
# Ctrl+C pour arrêter, ou 'tilt down' pour nettoyer
```

**Voir [TILT.md](TILT.md) pour plus de détails.**

---

## 🚢 Déploiement Production

### Kubernetes + ArgoCD

**Cluster** : microk8s sur serveur dédié

**Architecture** :
```
Namespace: dagster
├── dagster-webserver       # UI Dagster (https://dagster.tgu.ovh)
├── dagster-daemon          # Scheduler + sensors
└── dagster-user-deployment # User code (assets bronze/silver/gold)

Namespace: datalab
└── postgresql              # DuckLake catalog

Namespace: ia-lab
└── rustfs-svc              # S3 compatible (https://rustfs.tgu.ovh)
```

**Déploiement** :

1. **Push vers main** déclenche GitHub Actions (CI/CD)
2. **CI Build** crée les images Docker
3. **CD Update Manifests** met à jour K8s YAML
4. **ArgoCD** détecte le changement et déploie automatiquement

**Voir [docs/deployment.md](docs/deployment.md) pour le guide complet.**

---

## 📊 Performances et Métriques

### Données

- **Période** : 2019-2024 (5 ans)
- **Zone** : Nouvelle-Calédonie (département 988)
- **Accidents positifs** : ~1000 après filtrage
- **Échantillons négatifs** : ~2200 (ratio 2.2:1)
- **Features** : 24 (base + enrichies)

### Modèles

**Architecture ensemble** : Blending de 3 modèles (poids: 0.4, 0.4, 0.2)

1. **CatBoost** (gradient boosting)
   - Optimisation : Optuna (50 trials)
   - Gestion native des catégorielles
   - Meilleur AUC individuel

2. **XGBoost** (gradient boosting)
   - Régularisation L1/L2
   - Très rapide en inférence
   - Complémentaire à CatBoost

3. **MLP** (réseau de neurones)
   - Embeddings catégoriels
   - Capture relations non-linéaires
   - Diversité d'approche

**Performances ensemble** :
- **Métrique principale** : Recall (priorité détection accidents)
- **Recall** : **92.2%**
- **Precision** : 94.5%
- **F1-Score** : 93.3%
- **AUC-ROC** : **98.0%** (Blend) vs 97.8% (CatBoost) / 97.7% (XGBoost) / 97.3% (MLP)
- **Seuil optimal** : 0.641 (optimisé sur la courbe Precision-Recall)

### Rapports de Qualité (Evidently AI)

Le pipeline inclut désormais une étape de reporting automatisée via **Evidently AI** (asset `evidently_report`).
Ce rapport génère un dashboard HTML interactif contenant :
- **Classification Dashboard** : ROC, PR curves, et métriques par seuil.
- **Data Drift** : Détection de décalage statistique entre les données de train (référence) et le jeu de test actuel.
- **Feature Stats** : Statistiques descriptives complètes des caractéristiques.
- **Contextualisation** : Accompagné de textes explicatifs pour faciliter l'interprétation des résultats.

### Features Importantes

1. **hour_of_day** (heure)
2. **geo_cluster** (position regroupée par K-Means)
3. **atm** (conditions météo)
4. **day_of_week** (jour semaine)
5. **road_type_id** (type de route)

---

## 🤝 Contribution

### Workflow Git

```bash
# 1. Créer une branche feature
git checkout -b feature/ma-nouvelle-feature

# 2. Développer et tester
# ... modifications code ...
uv run pytest tests/unit/ -v
uv run ruff check src/

# 3. Commit et push
git add .
git commit -m "feat: ajout feature X"
git push origin feature/ma-nouvelle-feature

# 4. Ouvrir Pull Request sur GitHub
# → CI Tests s'exécute automatiquement
# → Review + merge vers main
# → CD Deploy automatique
```

### Conventions

- **Commits** : [Conventional Commits](https://www.conventionalcommits.org/)
  - `feat:` nouvelle fonctionnalité
  - `fix:` correction bug
  - `docs:` documentation
  - `refactor:` refactoring code
  - `test:` ajout tests
  - `chore:` maintenance

- **Code** :
  - Linter : ruff (ligne 100 caractères)
  - Tests : pytest (>80% coverage)
  - Docstrings : Google style

---

## 📄 Licence

MIT License - Voir [LICENSE](LICENSE)

---

## 📧 Contact

**Author** : Tom
**Project** : https://github.com/tom333/accidents-nc
**Dagster UI** : https://dagster.tgu.ovh
**Streamlit App** : https://streamlit.tgu.ovh
**MLflow Tracking** : (Artefacts de rapport Evidently disponibles dans les runs)

---

---

## 🙏 Remerciements

- **data.gouv.fr** : Données accidents officielles
- **OpenStreetMap** : Réseau routier
- **Dagster** : Orchestration pipeline
- **DuckDB** : Moteur SQL analytique
- **CatBoost** : Modèle ML performant
