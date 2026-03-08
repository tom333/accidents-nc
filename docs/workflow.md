# 🔄 Workflow de Développement - Accidents NC

Ce guide détaille le workflow de développement quotidien pour contributeurs et data scientists.

---

## 📋 Table des Matières

1. [Setup Initial](#setup-initial)
2. [Développement Local](#développement-local)
3. [Développement avec Tilt](#développement-avec-tilt)
4. [Travailler avec Dagster](#travailler-avec-dagster)
5. [Exploration avec Marimo](#exploration-avec-marimo)
6. [Tests](#tests)
7. [Workflow Git](#workflow-git)
8. [CI/CD](#cicd)
9. [Troubleshooting](#troubleshooting)

---

## 🚀 Setup Initial

### 1. Cloner le Projet

```bash
git clone https://github.com/tom333/accidents-nc.git
cd accidents
```

### 2. Installer uv (gestionnaire de paquets)

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ou via pip
pip install uv
```

**Pourquoi uv ?**
- 10-100x plus rapide que pip
- Gestion de lockfile automatique (uv.lock)
- Résolution de dépendances optimisée
- Compatible avec pyproject.toml

### 3. Installer les Dépendances

```bash
# Production seulement
uv sync

# Production + Training (CatBoost, Optuna, etc.)
uv sync --extra training

# Production + Dev (Marimo, pytest, ruff)
uv sync --extra dev

# Tout installer
uv sync --all-extras
```

### 4. Activer l'Environnement

```bash
source .venv/bin/activate

# Vérifier l'installation
python --version  # Python 3.13+
dagster --version # dagster 1.8+
```

### 5. Configuration Environnement

Copier `.env.example` vers `.env` :

```bash
cp .env.example .env
```

Éditer `.env` avec vos credentials (DuckLake prod) :

```bash
# DuckLake PostgreSQL catalog
POSTGRES_URL=postgresql://accidents:***@postgres.datalab.svc.cluster.local:5432/accidents

# S3 Storage (RustFS)
S3_ENDPOINT=https://rustfs.tgu.ovh
S3_BUCKET=accidents
AWS_ACCESS_KEY_ID=***
AWS_SECRET_ACCESS_KEY=***

# Dagster (optionnel pour prod)
DAGSTER_POSTGRES_USER=dagster
DAGSTER_POSTGRES_PASSWORD=***
DAGSTER_POSTGRES_DB=dagster
```

---

## 💻 Développement Local

### Option A : Infrastructure Docker Compose (recommandé)

Démarre PostgreSQL + MinIO S3 localement pour simuler DuckLake :

```bash
cd infra/docker
docker-compose -f docker-compose.dev.yml up -d

# Vérifier les services
docker-compose ps

# Logs
docker-compose logs -f postgres
docker-compose logs -f minio
```

**Services démarrés** :
- PostgreSQL : `localhost:5432`
- MinIO S3 : `localhost:9000` (API) + `localhost:9001` (Console)

**Accéder à MinIO Console** :
- URL : http://localhost:9001
- Login : `minioadmin` / `minioadmin`
- Bucket `accidents` créé automatiquement

### Option B : Services Externes

Utiliser PostgreSQL et S3 de production (mettre les bons credentials dans `.env`).

### Lancer Dagster UI Localement

```bash
cd /path/to/accidents
export PYTHONPATH=$PWD
dagster dev -f src/definitions.py

# Ouvrir http://localhost:3000
```

**Interface Dagster UI** :
- Assets : Voir tous les assets (bronze/silver/gold)
- Materialize : Exécuter un asset
- Runs : Historique des exécutions
- Schedules : Jobs planifiés
- Sensors : Triggers événementiels

---

## ⚡ Développement avec Tilt

**Tilt** permet un **live-reload ultra-rapide** (2-5 secondes) sans rebuild Docker complet.

### Pourquoi Tilt ?

| Méthode | Rebuild | Redéploiement | Total |
|---------|---------|---------------|-------|
| **rebuild-and-deploy.sh** | 1min 30s | 30s | **2 minutes** |
| **Tilt** | - | 2-5s | **2-5 secondes** |

Tilt sync le code Python directement dans le pod sans rebuild image !

### Installation Tilt

```bash
# Linux/macOS
curl -fsSL https://raw.githubusercontent.com/tilt-dev/tilt/master/scripts/install.sh | bash

# Ou via Homebrew
brew install tilt
```

### Démarrer Tilt

```bash
cd /path/to/accidents
tilt up

# Interface web : http://localhost:10350
```

**Interface Tilt** :
- **dagster-user** : Logs du pod Dagster
- **Sync status** : Fichiers synchronisés
- **Errors** : Erreurs build/runtime

### Workflow avec Tilt

```bash
# 1. Démarrer Tilt
tilt up

# 2. Modifier du code Python
vim src/bronze/ingest.py

# 3. Tilt détecte automatiquement et sync (~2s)
# Pas besoin de rebuild !

# 4. Vérifier dans Dagster UI
# Ouvrir http://localhost:3000 (ou dagster.tgu.ovh)
# Matérialiser l'asset bronze_accidents_nc

# 5. Arrêter Tilt
# Ctrl+C ou 'tilt down'
```

### Fichiers Synchronisés

[Tiltfile](../Tiltfile) :

```python
sync(
    'dagster-user',
    sync_path='.',
    target_path='/opt/dagster/app',
    ignore=[
        '.venv',
        '__pycache__',
        '*.pyc',
        'data/',
        'mlruns/',
    ]
)
```

Tous les fichiers Python sont synchronisés sauf :
- `.venv/` (environnement virtuel)
- `__pycache__/` (cache Python)
- `data/` (trop gros)
- `mlruns/` (artefacts MLflow)

**Voir [TILT.md](../TILT.md) pour guide complet.**

---

## 🎭 Travailler avec Dagster

### Structure Assets

```
src/assets/
├── bronze/             # Assets bronze
├── silver/             # Assets silver
└── gold/               # Assets gold
src/definitions.py      # Définitions Dagster (entry point)
src/schedules.py        # Jobs schedulés
src/resources/
└── ducklake.py         # Resource DuckLake common
```

### Créer un Nouvel Asset

**Exemple** : Asset pour générer des prédictions quotidiennes

```python
# src/assets/predictions.py

from dagster import asset, AssetExecutionContext
from src.utils.predictions import generate_daily_predictions

@asset(
    deps=["gold_models"],  # Dépend de gold_models
    group_name="predictions",
    compute_kind="python"
)
def daily_predictions(context: AssetExecutionContext) -> int:
    """Génère les prédictions pour les 24 prochaines heures."""

    context.log.info("Génération prédictions quotidiennes...")

    count = generate_daily_predictions()

    context.log.info(f"✅ {count} prédictions générées")
    return count
```

**Enregistrer l'asset** dans# src/definitions.py :

```python
from dagster import Definitions
from src.assets.bronze import bronze_accidents_nc
from src.assets.silver import silver_features
from src.assets.gold import gold_train_test, gold_models
from src.assets.predictions import daily_predictions  # NOUVEAU

defs = Definitions(
    assets=[
        bronze_accidents_nc,
        silver_features,
        gold_train_test,
        gold_models,
        daily_predictions,  # NOUVEAU
    ],
)
```

### Matérialiser un Asset

**Via UI Dagster** :
1. Ouvrir http://localhost:3000
2. Onglet "Assets"
3. Sélectionner `daily_predictions`
4. Cliquer "Materialize"

**Via CLI** :
```bash
dagster asset materialize -m src.definitions -a daily_predictions
```

**Via Code** :
```python
from dagster import materialize
from src.assets.predictions import daily_predictions

result = materialize([daily_predictions])
```

### Tester un Asset Localement

```python
# test_predictions.py

from dagster import build_asset_context
from src.assets.predictions import daily_predictions

def test_daily_predictions():
    context = build_asset_context()
    result = daily_predictions(context)
    assert result > 0
```

### Ajouter un Schedule

```python
# src/schedules.py

from dagster import ScheduleDefinition, define_asset_job

# Job qui exécute daily_predictions
predictions_job = define_asset_job(
    name="predictions_job",
    selection=["daily_predictions"]
)

# Schedule tous les jours à 6h
daily_predictions_schedule = ScheduleDefinition(
    job=predictions_job,
    cron_schedule="0 6 * * *",  # 6h tous les jours
)
```

Enregistrer dans `definitions.py` :

```python
defs = Definitions(
    assets=[...],
    schedules=[daily_predictions_schedule],
)
```

---

## 🔬 Exploration avec Marimo

**Marimo** = Notebooks réactifs Python (alternative moderne à Jupyter).

### Pourquoi Marimo ?

| Feature | Jupyter | Marimo |
|---------|---------|--------|
| Réactivité | ❌ Manuelle | ✅ Automatique |
| Reproductibilité | ⚠️ Order-dependent | ✅ DAG déterministe |
| Git-friendly | ❌ JSON illisible | ✅ Pure Python |
| UI Elements | ⚠️ ipywidgets | ✅ Built-in (sliders, buttons) |

### Notebooks Disponibles

```
exploration/
├── 01_eda_accidents.py           # EDA accidents NC
├── 02_features_engineering.py    # Features temporelles/spatiales
├── 03_model_experiments.py       # Training modèles
└── 04_viz_predictions.py         # Visualisation prédictions
```

### Lancer un Notebook Marimo

```bash
# Mode édition (interactif)
marimo edit exploration/01_eda_accidents.py

# Mode lecture seule (app)
marimo run exploration/04_viz_predictions.py

# Ouvrir http://localhost:2718
```

### Créer un Nouveau Notebook

```bash
marimo edit exploration/05_my_analysis.py
```

**Template** :

```python
import marimo

__generated_with = "0.9.0"
app = marimo.App()

@app.cell
def _():
    import marimo as mo
    import polars as pl
    from src.ducklake import get_client
    return mo, pl, get_client

@app.cell
def _(get_client):
    # Charger données depuis DuckLake
    client = get_client()
    df = client.table("bronze.accidents_nc").pl()
    return df,

@app.cell
def _(df, mo):
    # Afficher statistiques
    mo.md(f"""
    # Accidents NC

    - **Total accidents** : {len(df)}
    - **Période** : {df['date_accident'].min()} → {df['date_accident'].max()}
    """)
    return

@app.cell
def _(df):
    # Visualisation
    import altair as alt

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('month:O', title='Mois'),
        y=alt.Y('count()', title='Nombre accidents')
    )
    chart
    return

if __name__ == "__main__":
    app.run()
```

### Convertir Jupyter → Marimo

```bash
marimo convert notebook.ipynb -o exploration/notebook.py
```

---

## 🧪 Tests

### Structure Tests

```
tests/
├── unit/                       # Tests unitaires (mocks)
│   ├── test_bronze.py          # Tests ingestion
│   ├── test_silver.py          # Tests features
│   └── test_gold.py            # Tests datasets + training
├── integration/                # Tests intégration (DuckLake réel)
│   └── test_pipeline_e2e.py
└── test_phase*_validation.py   # Tests validation phases
```

### Lancer les Tests

```bash
# Tous les tests unitaires
uv run pytest tests/unit/ -v

# Test spécifique
uv run pytest tests/unit/test_bronze.py::TestIngestCaracteristiques::test_schema -v

# Tests avec coverage
uv run pytest tests/unit/ --cov=src --cov-report=html

# Voir coverage HTML
open htmlcov/index.html

# Tests intégration (marqués @pytest.mark.integration)
uv run pytest tests/integration/ -v

# Tests validation phases
python tests/test_phase1_validation.py
python tests/test_phase6_validation.py
```

### Écrire un Test Unitaire

```python
# tests/unit/test_my_module.py

import pytest
from unittest.mock import Mock, patch
from src.my_module import my_function

class TestMyFunction:

    def test_basic_behavior(self):
        """Test comportement de base."""
        result = my_function(input_data)
        assert result == expected_output

    @patch('src.my_module.get_client')
    def test_with_mock(self, mock_client):
        """Test avec mock DuckLake."""
        mock_client.return_value.table.return_value.pl.return_value = mock_df

        result = my_function()

        assert result > 0
        mock_client.assert_called_once()
```

### Linter (ruff)

```bash
# Vérifier le code
uv run ruff check src/ tests/

# Auto-fix
uv run ruff check src/ tests/ --fix

# Formater
uv run ruff format src/ tests/
```

---

## 🌳 Workflow Git

### Branches

```
main                # Production (protected)
├── develop         # Intégration (optionnel)
├── feature/xxx     # Nouvelles fonctionnalités
├── fix/xxx         # Corrections bugs
└── docs/xxx        # Documentation
```

### Workflow Feature Branch

```bash
# 1. Créer une branche depuis main
git checkout main
git pull origin main
git checkout -b feature/add-weather-features

# 2. Développer
vim src/silver/features.py
uv run pytest tests/unit/test_silver.py -v

# 3. Commit (Conventional Commits)
git add src/silver/features.py
git commit -m "feat(silver): add weather features from OpenWeather API"

# 4. Push
git push origin feature/add-weather-features

# 5. Ouvrir Pull Request sur GitHub
# → CI Tests s'exécute automatiquement
# → Review par les mainteneurs
# → Merge vers main
```

### Conventions de Commit

**Format** : `<type>(<scope>): <description>`

**Types** :
- `feat:` nouvelle fonctionnalité
- `fix:` correction bug
- `docs:` documentation
- `refactor:` refactoring sans changement fonctionnel
- `test:` ajout/modification tests
- `chore:` maintenance (deps, config)
- `perf:` amélioration performance
- `style:` formatage code

**Exemples** :
```bash
feat(bronze): add new data source from OSM
fix(gold): correct stratified split ratio
docs(readme): update installation instructions
refactor(silver): extract negatives sampling to separate function
test(gold): add unit tests for training pipeline
chore(deps): update dagster to 1.9.0
```

### Revue de Code

**Checklist PR** :
- [ ] Code respecte conventions (ruff)
- [ ] Tests unitaires passent (pytest)
- [ ] Coverage > 80%
- [ ] Documentation mise à jour
- [ ] Commit messages suivent conventions
- [ ] Pas de secrets commités

---

## 🚀 CI/CD

### Workflow GitHub Actions

```
Push → CI Tests → CI Build → CD Update Manifests → ArgoCD Deploy
```

### CI Tests ([.github/workflows/ci-tests.yml](../.github/workflows/ci-tests.yml))

**Triggers** :
- Push sur `main` ou `develop`
- Pull requests vers `main`

**Jobs** :
1. **lint** : ruff check
2. **unit-tests** : pytest + coverage
3. **validation-tests** : tests phases 1-6

**Durée** : ~5 minutes

### CI Build ([.github/workflows/ci-build.yml](../.github/workflows/ci-build.yml))

**Triggers** :
- Push sur `main`
- Tags `v*` (releases)
- Manual dispatch

**Jobs** :
1. **build-dagster** : Build Dockerfile.dagster
2. **build-streamlit** : Build Dockerfile.streamlit
3. **summary** : Résumé builds

**Features** :
- Docker Buildx (multi-platform)
- GitHub Actions cache (accélère builds)
- Trivy security scan
- Tags automatiques (sha, branch, latest)

**Durée** : ~10 minutes

### CD Update Manifests ([.github/workflows/cd-update-manifests.yml](../.github/workflows/cd-update-manifests.yml))

**Triggers** :
- Après CI Build réussi
- Manual dispatch

**Steps** :
1. Checkout repo
2. Update image tags dans K8s YAML (sed)
3. Git commit + push vers main
4. ArgoCD détecte changement et déploie

**Durée** : ~1 minute

### ArgoCD Auto-Sync

**Config** : [infra/k8s/dagster/argocd-application.yaml](../infra/k8s/dagster/argocd-application.yaml)

```yaml
syncPolicy:
  automated:
    prune: true       # Supprime ressources obsolètes
    selfHeal: true    # Répare divergences
```

**Polling** : 3 minutes

**Durée déploiement** : ~5 minutes

**Total CI/CD** : ~22 minutes (push → deployed)

---

## 🛠️ Troubleshooting

### Problème : Dagster ne démarre pas

**Symptômes** :
```
dagster.core.errors.DagsterInvariantViolationError: Could not load repository
```

**Solutions** :
```bash
# 1. Vérifier PYTHONPATH
export PYTHONPATH=$PWD
echo $PYTHONPATH

# 2. Vérifier imports
python -c "from src.definitions import defs; print(defs)"

# 3. Vérifier dépendances
uv sync
source .venv/bin/activate
```

### Problème : DuckLake connection error

**Symptômes** :
```
psycopg2.OperationalError: could not connect to server
```

**Solutions** :
```bash
# 1. Vérifier PostgreSQL
docker-compose ps postgres
docker-compose logs postgres

# 2. Vérifier .env
cat .env | grep POSTGRES

# 3. Tester connexion
python -c "from src.ducklake import get_client; client = get_client(); print(client)"
```

### Problème : S3 Access Denied

**Symptômes** :
```
botocore.exceptions.ClientError: An error occurred (403) when calling the PutObject operation: Forbidden
```

**Solutions** :
```bash
# 1. Vérifier MinIO
open http://localhost:9001
# Login: minioadmin / minioadmin

# 2. Vérifier bucket existe
docker exec -it minio mc ls local/

# 3. Créer bucket si nécessaire
docker exec -it minio mc mb local/accidents

# 4. Vérifier credentials .env
cat .env | grep S3
```

### Problème : Tests échouent

**Symptômes** :
```
ImportError: No module named 'src.accidents'
```

**Solutions** :
```bash
# 1. Installer en mode editable
uv pip install -e .

# 2. Vérifier PYTHONPATH
export PYTHONPATH=$PWD
pytest tests/unit/ -v

# 3. Vérifier mocks
# Les tests unitaires doivent mocker DuckLake, pas utiliser vraie DB
```

### Problème : Tilt ne sync pas

**Symptômes** :
```
Tilt: No changes detected
```

**Solutions** :
```bash
# 1. Vérifier Tiltfile ignore patterns
cat Tiltfile | grep ignore

# 2. Forcer rebuild
tilt trigger dagster-user

# 3. Redémarrer Tilt
tilt down
tilt up
```

### Problème : ArgoCD out of sync

**Symptômes** :
```
ArgoCD UI: Status "OutOfSync"
```

**Solutions** :
```bash
# 1. Vérifier différences
kubectl get application dagster-accidents-pipeline -n argocd -o yaml

# 2. Sync manuellement
argocd app sync dagster-accidents-pipeline

# 3. Hard refresh
argocd app sync dagster-accidents-pipeline --force
```

### Logs Utiles

```bash
# Logs Dagster
kubectl logs -n dagster deployment/dagster-user-deployment -f

# Logs PostgreSQL
docker-compose logs -f postgres

# Logs MinIO
docker-compose logs -f minio

# Logs ArgoCD
kubectl logs -n argocd deployment/argocd-server -f
```

---

## 📚 Ressources

### Documentation
- [README.md](../README.md) : Vue d'ensemble projet
- [docs/architecture.md](architecture.md) : Architecture détaillée
- [docs/deployment.md](deployment.md) : Guide déploiement
- [TILT.md](../TILT.md) : Guide Tilt détaillé
- [INDUSTRIALISATION.md](../INDUSTRIALISATION.md) : Historique migration

### Liens Externes
- [Dagster Docs](https://docs.dagster.io/)
- [Marimo Docs](https://docs.marimo.io/)
- [uv Docs](https://docs.astral.sh/uv/)
- [Tilt Docs](https://docs.tilt.dev/)
- [ArgoCD Docs](https://argo-cd.readthedocs.io/)

### Support
- **Issues GitHub** : https://github.com/tom333/accidents-nc/issues
- **Discussions** : https://github.com/tom333/accidents-nc/discussions
