# Déploiement Dagster sur Kubernetes

Guide de déploiement du pipeline accidents avec Dagster user-deployment.

> **Note** : Si vous utilisez ArgoCD, consultez [ARGOCD.md](ARGOCD.md) pour un déploiement GitOps automatisé.

## Prérequis

- Cluster Kubernetes avec Dagster webserver + daemon déjà déployés
- Service PostgreSQL (`postgresql.datalab.svc.cluster.local`) pour catalogue DuckLake
- Service RustFS (`rustfs-svc.ia-lab.svc.cluster.local`) pour stockage S3
- `kubectl` configuré
- Docker ou Microk8s pour build d'images

## Architecture déployée

```
┌─────────────────────────────────────────────────────┐
│ Dagster Webserver + Daemon (namespace default)      │
│   ↓ gRPC                                            │
│ Dagster User-Code: accidents_pipeline               │
│   └─ Assets: bronze → silver → gold                 │
│      ├─ bronze_accidents_nc                         │
│      ├─ silver_features                             │
│      ├─ gold_datasets                               │
│      └─ gold_models                                 │
└─────────────────────────────────────────────────────┘
           ↓                          ↓
    ┌─────────────┐          ┌─────────────────┐
    │ PostgreSQL  │          │ RustFS (S3)     │
    │ (datalab)   │          │ (ia-lab)        │
    │ Catalogue   │          │ Fichiers Parquet│
    │ DuckLake    │          │ DuckLake        │
    └─────────────┘          └─────────────────┘
```

## Étapes de déploiement

### 1. Configuration des credentials

Éditez [k8s/dagster/configmap-dagster-ducklake.yaml](configmap-dagster-ducklake.yaml) :

```yaml
# Dans le Secret dagster-ducklake-secrets :
POSTGRES_PASSWORD: "votre_mot_de_passe"
AWS_ACCESS_KEY_ID: "votre_access_key"
AWS_SECRET_ACCESS_KEY: "votre_secret_key"
```

### 2. Lancement du déploiement

```bash
./k8s/dagster/deploy-dagster-user.sh
```

Ce script :
1. Build l'image Docker `accidents-dagster:latest`
2. Importe l'image dans microk8s (ou push vers registry)
3. Crée le PVC pour les modèles (5Gi)
4. Applique ConfigMap/Secret DuckLake
5. Déploie le pod user-code avec Service
6. Attend que le pod soit ready

### 3. Configuration du workspace Dagster

Le fichier [workspace.yaml](workspace.yaml) doit être intégré dans la configuration du webserver Dagster.

**Option A : Via ConfigMap existante**

Si Dagster est déployé via Helm :

```bash
# Récupérer la ConfigMap actuelle
kubectl get configmap dagster-instance -o yaml > /tmp/dagster-config.yaml

# Éditer et ajouter workspace.yaml
kubectl edit configmap dagster-instance
```

Ajouter dans `data` :

```yaml
data:
  workspace.yaml: |
    load_from:
      - grpc_server:
          host: dagster-user-code-accidents
          port: 4000
          location_name: "accidents_pipeline"
```

**Option B : Modifier le Helm values.yaml**

```yaml
dagsterWebserver:
  workspace:
    enabled: true
    servers:
      - host: dagster-user-code-accidents
        port: 4000
        location_name: accidents_pipeline
```

### 4. Redémarrer le webserver

```bash
kubectl rollout restart deployment dagster-webserver
kubectl wait --for=condition=ready pod -l app=dagster-webserver --timeout=120s
```

### 5. Vérification

```bash
# Pods user-code
kubectl get pods -l app=dagster-user-code,component=accidents

# Logs
kubectl logs -f -l app=dagster-user-code,component=accidents

# Services
kubectl get svc | grep dagster-user-code-accidents
```

Accédez à l'UI Dagster et vérifiez :
- Le workspace "accidents_pipeline" apparaît
- Les 4 assets bronze/silver/gold sont visibles
- Le graphe de dépendances est correct

### 6. Test de matérialisation

Dans l'UI Dagster :
1. Accédez à l'onglet "Assets"
2. Sélectionnez `gold_models`
3. Cliquez "Materialize"
4. Dagster exécutera automatiquement la chaîne complète :
   - `bronze_accidents_nc` (ingestion data.gouv.fr)
   - `silver_features` (enrichissement OSM + négatifs)
   - `gold_datasets` (train/test split)
   - `gold_models` (entraînement multi-modèles)

## Surveillance et debugging

### Logs en temps réel

```bash
kubectl logs -f deployment/dagster-user-deployment-accidents
```

### Inspection du pod

```bash
kubectl exec -it deployment/dagster-user-deployment-accidents -- bash
# Vérifier les imports Python
python -c "from dagster_accidents.repository import defs; print(defs)"
```

### Vérifier la connectivité DuckLake

```bash
# Depuis le pod
kubectl exec -it deployment/dagster-user-deployment-accidents -- bash
python -c "
from pipeline.config import ensure_connection
conn = ensure_connection()
print(conn.execute('SELECT current_database()').fetchall())
"
```

### Problèmes courants

**Pod CrashLoopBackOff** :
- Vérifier les logs : `kubectl logs deployment/dagster-user-deployment-accidents`
- Vérifier l'image existe : `microk8s ctr images ls | grep accidents-dagster`

**Workspace non détecté** :
- Vérifier que workspace.yaml est bien monté dans webserver
- Vérifier connectivité : `kubectl exec deployment/dagster-webserver -- curl dagster-user-code-accidents:4000`

**Erreurs DuckLake** :
- Vérifier les variables d'environnement : `kubectl get configmap dagster-ducklake-config -o yaml`
- Tester connectivité Postgres : `kubectl exec deployment/dagster-user-deployment-accidents -- nc -zv postgresql.datalab.svc.cluster.local 5432`
- Tester connectivité RustFS : `kubectl exec deployment/dagster-user-deployment-accidents -- curl http://rustfs-svc.ia-lab.svc.cluster.local:9000`

## Mise à jour du code

```bash
# Rebuild l'image
docker build -f Dockerfile.dagster -t accidents-dagster:latest .
docker save accidents-dagster:latest | microk8s ctr image import -

# Redémarrer le deployment
kubectl rollout restart deployment/dagster-user-deployment-accidents
```

## Nettoyage

```bash
kubectl delete -f k8s/dagster/deployment-dagster-user.yaml
kubectl delete -f k8s/dagster/configmap-dagster-ducklake.yaml
kubectl delete -f k8s/dagster/pvc-dagster-models.yaml
```

## Variables d'environnement

Toutes les variables DuckLake sont injectées via ConfigMap/Secret :

| Variable | Source | Description |
|----------|--------|-------------|
| `DUCKLAKE_DATABASE_URL` | ConfigMap | DSN Postgres catalogue |
| `DUCKLAKE_DATA_PATH` | ConfigMap | Chemin S3 fichiers Parquet |
| `DUCKLAKE_DB_ALIAS` | ConfigMap | Alias base DuckLake |
| `AWS_ENDPOINT_URL` | ConfigMap | Endpoint RustFS |
| `AWS_REGION` | ConfigMap | Région S3 |
| `AWS_ACCESS_KEY_ID` | Secret | Credentials RustFS |
| `AWS_SECRET_ACCESS_KEY` | Secret | Credentials RustFS |

## Prochaines étapes

1. **Schedules Dagster** : créer un schedule pour exécution quotidienne
2. **Monitoring** : intégrer Prometheus metrics Dagster
3. **Alertes** : configurer Slack/email pour échecs de runs
4. **Scaling** : ajuster resources requests/limits selon charge
