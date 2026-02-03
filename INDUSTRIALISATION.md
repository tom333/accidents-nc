# Industrialisation avec Dagster sur Kubernetes local

Ce document décrit les étapes pour industrialiser le pipeline d'accidents avec **Dagster**
sur un cluster **Kubernetes local**, en s'appuyant sur **DuckLake** pour le stockage
des données (moteur DuckDB managé) et les composants suivants :

- Une instance Dagster (Dagster webserver + daemon) est déjà déployée sur le cluster.
- La configuration utilise `dagster-user-deployments`.
- Une base Postgres accessible via le service `postgresql` dans le namespace `datalab`,
  utilisée comme métastore DuckLake.
- Un service compatible S3 (RustFS) accessible via le service `rustfs-svc` dans le
  namespace `ia-lab`, utilisé comme stockage objet pour DuckLake.
- Un volume (PersistentVolume / hostPath) est monté dans le pod user-deployment et expose
  le code local du projet `accidents`.

## Architecture Médaillons (Bronze/Silver/Gold)

Le pipeline est organisé selon l'architecture **médaillons** standard des lakehouses :

### 🥉 Bronze Layer (Données brutes)
- **Schéma** : `bronze`
- **Tables** : 
  - `bronze.caracteristiques` : CSV bruts data.gouv.fr 2019-2024
  - `bronze.usagers` : CSV usagers data.gouv.fr
  - `bronze.accidents_nc` : accidents New Caledonia (dep=988) jointure carac+usagers
- **Asset Dagster** : `bronze_accidents_nc`
- **Transformation** : ingestion brute avec filtrage département, nettoyage coordonnées
- **Idempotence** : `CREATE OR REPLACE TABLE`

### 🥈 Silver Layer (Données enrichies)
- **Schéma** : `silver`
- **Tables** :
  - `silver.full_dataset` : features enrichies (positifs + négatifs synthétiques)
- **Asset Dagster** : `silver_features`
- **Transformations** :
  - Parsing datetime (jour/mois/an hrmn → features temporelles)
  - Enrichissement routier OSMnx (buffer 200m, comptage intersections)
  - Génération négatifs synthétiques (ratio 22k, sampling des marginales)
  - Features d'interaction (hour×weekend, month×dayofweek)
- **Qualité** : dédupliqué, valeurs nulles gérées, géolocalisation validée

### 🥇 Gold Layer (Datasets ML & Modèles)
- **Schéma** : `gold`
- **Tables** :
  - `gold.train` : dataset train (80%)
  - `gold.test` : dataset test (20%)
  - `gold.feature_metadata` : ordre et noms des features
- **Assets Dagster** : `gold_datasets`, `gold_models`
- **Artefacts** :
  - `atm_encoder.pkl` : LabelEncoder pour météo
  - `features.pkl` : liste des features pour prédiction
  - `accident_model.pkl` : meilleur modèle (RandomForest, LGBM, XGB, CatBoost, TabNet ou MLP)
- **MLflow** : tracking des runs, métriques AUC-ROC, hyperparamètres
- **Sélection** : RandomizedSearchCV 5-fold, optimisation AUC-ROC

## 1. Structurer le code pour Dagster ✅

1. Créer un package Dagster dédié (dans ce repo) : ✅ (fait : dossier `dagster_accidents/` + fichiers de base)
   - Dossier `dagster_accidents/` (ou `accidents_dagster/`) à la racine.
   - Fichiers minimum :
     - `dagster_accidents/__init__.py`
     - `dagster_accidents/jobs.py` : définition des jobs/pipelines.
   - `dagster_accidents/assets.py` (optionnel mais recommandé) : assets data (tables gérées par DuckLake / DuckDB).
     - `dagster_accidents/repository.py` : objet `Definitions` / `repository` Dagster.

2. Le code métier reste dans `pipeline/` comme **source de vérité** (✅ migration DuckLake dans `pipeline.config.ensure_connection`) :
   - `pipeline.stage_ingest.ingest_all` → écrit dans `bronze`
   - `pipeline.stage_features.build_feature_store` → écrit dans `silver`
   - `pipeline.stage_datasets.build_datasets` → écrit dans `gold`
   - `pipeline.stage_modeling.run_training` → lit `gold`, écrit artefacts + MLflow

3. Les ops/assets Dagster ne font que **appeler ces fonctions** en les décorant :
   - Exemple : `@op` ou `@asset` qui appelle `ingest_all()` et renvoie des métriques (dict).

4. Ajouter Dagster dans les dépendances : ✅ (fait dans `pyproject.toml`)
   - Dans `pyproject.toml` : `dagster`, `dagster-webserver`, `dagster-k8s` (si besoin).

## 2. Modéliser le pipeline dans Dagster

Deux options principales :

### 2.1. Modélisation en jobs (simple)

1. Créer un fichier `dagster_accidents/jobs.py` avec :
   - Un `@job` principal (ex. `accidents_full_job`) qui enchaîne 4 `@op` :
     - `ingest_raw_op` → appelle `ingest_all`.
     - `build_features_op` → appelle `build_feature_store`.
     - `prepare_datasets_op` → appelle `build_datasets`.
     - `train_models_op` → appelle `run_training`.

2. Chaque `@op` :
   - Loggue des infos importantes (nb de lignes, AUC, etc.).
   - Renvoie un petit résumé (dict) pour inspection éventuelle.

3. Avantage :
   - Très proche de l'orchestrateur Python existant.
   - Facile à déclencher via la Dagster UI ou cron Dagster.

### 2.2. Modélisation en assets (recommandée pour la data) ✅

1. Créer un fichier `dagster_accidents/assets.py` avec des `@asset` représentant les médaillons DuckLake : ✅
   - `bronze_accidents_nc` (key_prefix=["bronze"], group_name="bronze") → écrit `bronze.accidents_nc` via `ingest_all`.
   - `silver_features` (key_prefix=["silver"], group_name="silver") → écrit `silver.full_dataset` via `build_feature_store`.
   - `gold_datasets` (key_prefix=["gold"], group_name="gold") → écrit `gold.train` / `gold.test` via `build_datasets`.
   - `gold_models` (key_prefix=["gold"], group_name="gold") → écrit artefacts ML + MLflow via `run_training`.

2. Décrire les dépendances via `deps` ou en utilisant les arguments des assets : ✅
   - `silver_features` dépend de `bronze_accidents_nc`.
   - `gold_datasets` dépend de `silver_features`.
   - `gold_models` dépend de `gold_datasets`.

3. Avantages :
   - Vue `Asset Graph` très claire dans la Dagster UI avec groupes bronze/silver/gold.
   - Rebuild partiel : possibilité de ne relancer qu'un asset obsolète.
   - Traçabilité du lineage des données (bronze → silver → gold).

## 3. Définir le repository / Definitions ✅

1. Créer `dagster_accidents/repository.py` avec : ✅ (fait : objet `defs` de type `Definitions`)
   - Soit un `Definitions` (Dagster >=1.5) :
     - `Definitions(assets=[...], jobs=[...], schedules=[...], sensors=[...])`.
   - Soit un `@repository` (ancienne API).

2. Y enregistrer : ✅ (fait : enregistrement des 4 assets bronze/silver/gold, sans jobs ni schedules pour l'instant)
   - Les jobs (`accidents_full_job`).
   - Ou/et les assets (`bronze_accidents_nc`, `silver_features`, `gold_datasets`, `gold_models`).
   - Les éventuels schedules (ex. job quotidien) pour remplacer le CronJob k8s ou le compléter.

3. Vérifier localement :
   - `uv run dagster dev -m dagster_accidents.repository` (ou équivalent) pour tester la définition.

## 4. Préparer le déploiement Dagster sur k8s (user-deployments) ✅

**Statut** : Fichiers de déploiement créés dans `k8s/dagster/`

### 4.1. Image Docker ✅

Créé [Dockerfile.dagster](Dockerfile.dagster) avec :
- Base Python 3.13-slim
- Dépendances système pour geospatial (GDAL, GEOS, PROJ)
- Installation via `uv` des packages ML/geo
- Copies des dossiers `pipeline/` et `dagster_accidents/`
- `PYTHONPATH=/opt/dagster/app`
- Exposition du port 4000 pour gRPC

### 4.2. Manifests Kubernetes ✅

Créés dans [k8s/dagster/](k8s/dagster/) :

1. **[configmap-dagster-ducklake.yaml](k8s/dagster/configmap-dagster-ducklake.yaml)** :
   - ConfigMap avec variables DuckLake (DATABASE_URL, DATA_PATH, endpoints)
   - Secret pour credentials (AWS S3, Postgres password)

2. **[deployment-dagster-user.yaml](k8s/dagster/deployment-dagster-user.yaml)** :
   - Deployment du user-code avec commande `dagster api grpc`
   - Injection des variables d'environnement depuis ConfigMap/Secret
   - Resources : 2Gi RAM request, 4Gi limit
   - Volume mount pour stockage modèles (PVC)
   - Service ClusterIP sur port 4000

3. **[pvc-dagster-models.yaml](k8s/dagster/pvc-dagster-models.yaml)** :
   - PersistentVolumeClaim 5Gi pour artefacts ML

4. **[workspace.yaml](k8s/dagster/workspace.yaml)** :
   - Configuration workspace Dagster pointant vers `dagster-user-code-accidents:4000`
   - Location name : `accidents_pipeline`

### 4.3. Script de déploiement automatisé ✅

Créé [k8s/dagster/deploy-dagster-user.sh](k8s/dagster/deploy-dagster-user.sh) qui :
1. Build l'image Docker `accidents-dagster:latest`
2. Import dans microk8s (ou push vers registry)
3. Applique les manifests k8s (PVC, ConfigMap, Deployment)
4. Attend que le pod soit ready
5. Affiche les logs et commandes utiles

Usage : `./k8s/dagster/deploy-dagster-user.sh`

### 4.4. Documentation détaillée ✅

Guide complet dans [k8s/dagster/DEPLOY_DAGSTER.md](k8s/dagster/DEPLOY_DAGSTER.md) :
- Architecture déployée (diagramme)
- Configuration des credentials
- Étapes de déploiement
- Configuration du workspace dans webserver Dagster
- Commandes de vérification et debugging
- Troubleshooting des problèmes courants

## 5. Intégration avec Kubernetes local

**Référence** : voir [k8s/dagster/DEPLOY_DAGSTER.md](k8s/dagster/DEPLOY_DAGSTER.md) pour la procédure complète.

### Étapes de déploiement :

1. **Éditer les credentials** dans `k8s/dagster/configmap-dagster-ducklake.yaml`

2. **Lancer le déploiement** :
   ```bash
   ./k8s/dagster/deploy-dagster-user.sh
   ```

3. **Configurer le workspace Dagster** :
   - Intégrer `k8s/dagster/workspace.yaml` dans la ConfigMap du webserver
   - Ou ajouter dans Helm values :
     ```yaml
     dagsterWebserver:
       workspace:
         servers:
           - host: dagster-user-code-accidents
             port: 4000
             location_name: accidents_pipeline
     ```

4. **Redémarrer le webserver** :
   ```bash
   kubectl rollout restart deployment dagster-webserver
   ```

5. **Vérification dans l'UI Dagster** :
   - Workspace "accidents_pipeline" visible
   - 4 assets bronze/silver/gold avec groupes
   - Graphe de dépendances correct

6. **Test de matérialisation** :
   - Sélectionner `gold_models` dans l'UI
   - Cliquer "Materialize"
   - Dagster exécute automatiquement : bronze → silver → gold

### Commandes de vérification :

```bash
# Status du pod
kubectl get pods -l app=dagster-user-code,component=accidents

# Logs en temps réel
kubectl logs -f -l app=dagster-user-code,component=accidents

# Tester connectivité DuckLake
kubectl exec deployment/dagster-user-deployment-accidents -- \
  python -c "from pipeline.config import ensure_connection; conn = ensure_connection(); print('OK')"
```

## 6. Programmation et supervision

1. Schedules Dagster :
   - Créer un `ScheduleDefinition` (ou entry `schedules=[...]` dans `Definitions`) pour :
     - Exécuter `accidents_full_job` chaque nuit / heure.
   - Activer le schedule dans l'UI.

2. Remplacer ou compléter les CronJobs k8s :
   - Option A : laisser Dagster gérer la planification (recommandé pour la cohérence des runs).
   - Option B : garder un CronJob k8s qui appelle l'API Dagster ou un job Dagster spécifique.

3. Monitoring :
   - Utiliser l'UI Dagster pour suivre les runs, relancer en cas d'échec.
   - Conserver MLflow pour le suivi expérimental des modèles (complémentaire à Dagster).

## 7. Rôle des notebooks marimo après migration

1. Les notebooks marimo restent des outils d'exploration et de visualisation :
   - Inspection des tables DuckLake (via un client DuckDB connecté à DuckLake) produites par Dagster.
   - Cartes Folium, analyses de features, SHAP, etc.

2. Ils ne sont plus responsables de l'orchestration :
   - Toute l'exécution planifiée se fait via Dagster.
   - Les notebooks peuvent éventuellement appeler des runs Dagster via l'API HTTP si besoin,
     mais ce n'est pas obligatoire.

3. Bien documenter dans README / INDUSTRIALISATION :
   - "Pipeline de production" = Dagster + k8s.
   - "Exploration / analyses ad hoc" = notebooks marimo.

## 8. Étapes pratiques de mise en œuvre (résumé)

### Phase 1 : Développement local ✅

1. ✅ Créer le package `dagster_accidents/` et y définir assets/jobs + repository
2. ✅ Ajouter Dagster aux dépendances du projet et tester localement (`dagster dev`)
3. ✅ Implémenter l'architecture médaillons (bronze/silver/gold)
4. ✅ Configurer `.env` avec variables DuckLake pour dev local

**Test local** : `uv run dagster dev -m dagster_accidents.repository`

### Phase 2 : Préparation k8s ✅

3. ✅ Créer `Dockerfile.dagster` pour l'image user-deployment
4. ✅ Créer les manifests k8s dans `k8s/dagster/`
   - ConfigMap/Secret DuckLake
   - Deployment + Service
   - PVC pour modèles
   - workspace.yaml
5. ✅ Créer script de déploiement automatisé
6. ✅ Documenter dans `k8s/dagster/DEPLOY_DAGSTER.md`

### Phase 3 : Déploiement et tests (à faire)

7. ⏳ Éditer credentials dans `k8s/dagster/configmap-dagster-ducklake.yaml`
8. ⏳ Lancer `./k8s/dagster/deploy-dagster-user.sh`
9. ⏳ Configurer workspace dans webserver Dagster
10. ⏳ Tester un run complet via l'UI sur le k8s local
11. ⏳ Ajouter schedules Dagster pour l'exécution récurrente
12. ⏳ Mettre à jour la documentation (README, STREAMLIT_APP, etc.) pour refléter l'architecture Dagster

### Organisation des fichiers k8s

Les manifests sont séparés par composant dans `k8s/` :

```
k8s/
├── README.md                    # Index et ordre de déploiement
├── dagster/                     # Orchestration ML
│   ├── DEPLOY_DAGSTER.md       # Guide complet
│   ├── configmap-dagster-ducklake.yaml
│   ├── deployment-dagster-user.yaml
│   ├── pvc-dagster-models.yaml
│   ├── workspace.yaml
│   └── deploy-dagster-user.sh
├── streamlit/                   # Application web
│   ├── README.md
│   ├── deployment.yaml
│   ├── service.yaml
│   └── deploy.sh
└── (fichiers communs)
    ├── namespace.yaml
    ├── cronjob-training.yaml
    └── setup-microk8s.sh
```
