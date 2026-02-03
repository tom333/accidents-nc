# Organisation des manifests Kubernetes

Ce dossier contient les configurations Kubernetes séparées par composant.

## Structure

```
k8s/
├── dagster/                    # Orchestration Dagster
│   ├── DEPLOY_DAGSTER.md      # Guide de déploiement complet
│   ├── configmap-dagster-ducklake.yaml
│   ├── deployment-dagster-user.yaml
│   ├── pvc-dagster-models.yaml
│   ├── workspace.yaml
│   └── deploy-dagster-user.sh
│
├── streamlit/                  # Application web de visualisation
│   ├── README.md
│   ├── deployment.yaml
│   ├── deployment-api.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── pvc.yaml
│   └── deploy.sh
│
├── namespace.yaml              # Namespace commun
├── cronjob-training.yaml       # CronJob pour entraînement (legacy)
├── deploy-mlflow.sh            # Déploiement MLflow
├── setup-microk8s.sh           # Configuration initiale cluster
└── KUBERNETES.md               # Documentation générale

```

## Composants

### Dagster (Orchestration)

Pipeline ML orchestré avec architecture médaillons (bronze/silver/gold).

**Déploiement** : voir [dagster/DEPLOY_DAGSTER.md](dagster/DEPLOY_DAGSTER.md)

### Streamlit (Visualisation)

Interface web pour explorer les prédictions et visualiser les cartes.

**Déploiement** : voir [streamlit/README.md](streamlit/README.md)

### Composants partagés

- `namespace.yaml` : namespace par défaut ou spécifique
- `cronjob-training.yaml` : entraînement périodique (peut être remplacé par schedules Dagster)
- `deploy-mlflow.sh` : déploiement du serveur MLflow pour tracking

## Ordre de déploiement recommandé

1. **Infrastructure de base**
   ```bash
   kubectl apply -f namespace.yaml
   ./setup-microk8s.sh  # Si premier déploiement
   ```

2. **MLflow (optionnel)**
   ```bash
   ./deploy-mlflow.sh
   ```

3. **Dagster orchestration**
   ```bash
   ./dagster/deploy-dagster-user.sh
   ```

4. **Streamlit app**
   ```bash
   ./streamlit/deploy.sh
   ```

## Dépendances externes

- PostgreSQL (namespace `datalab`) : catalogue DuckLake
- RustFS (namespace `ia-lab`) : stockage S3 DuckLake
- Dagster webserver + daemon : déjà déployés sur le cluster

## Variables d'environnement

Les credentials et configurations sont gérés via ConfigMap/Secret dans chaque dossier.

Voir :
- `dagster/configmap-dagster-ducklake.yaml` pour Dagster
- Variables d'environnement dans les deployments Streamlit
