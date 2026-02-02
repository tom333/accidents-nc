# Intégration ArgoCD pour Dagster User-Code

Ce guide explique comment déployer le user-code Dagster via ArgoCD.

## Prérequis

- Dagster déjà déployé via ArgoCD (Helm chart dans namespace `dagster`)
- Accès au cluster avec `kubectl` et `argocd` CLI
- PostgreSQL disponible : `postgresql.datalab.svc.cluster.local`
- RustFS disponible : `https://rustfs.tgu.ovh`

## Déploiement

### 1. Créer l'Application ArgoCD

Appliquez l'Application ArgoCD :

```bash
kubectl apply -f k8s/dagster/argocd-application.yaml
```

**Configuration de l'Application** :
- **Nom** : `dagster-accidents-pipeline`
- **Projet** : `infra-project`
- **Namespace** : `dagster`
- **Source** : `k8s/dagster/` de votre repo git
- **Sync Policy** : automatique avec prune et self-heal

### 2. Activer le workspace dans Dagster

Modifiez l'Application ArgoCD Dagster pour activer le workspace :

```bash
kubectl edit application dagster -n argocd
```

Dans la section `helm.values`, modifiez :

```yaml
dagsterWebserver:
  workspace:
    enabled: true
    servers:
      - host: dagster-user-code-accidents
        port: 4000
        location_name: accidents_pipeline
```

Ou appliquez cette configuration via un patch :

```bash
kubectl patch application dagster -n argocd --type merge -p '
spec:
  source:
    helm:
      values: |
        ingress:
          enabled: true
          annotations:
            cert-manager.io/cluster-issuer: "letsencrypt-prod"
            nginx.ingress.kubernetes.io/auth-url: "https://auth.tgu.ovh/oauth2/auth"
            nginx.ingress.kubernetes.io/auth-signin: "https://auth.tgu.ovh/oauth2/start?rd=https://dagster.tgu.ovh"
          dagsterWebserver:
            host: dagster.tgu.ovh
            path: /
            pathType: Prefix
            tls:
              enabled: true
              secretName: dagster-tls
        dagsterWebserver:
          workspace:
            enabled: true
            servers:
              - host: dagster-user-code-accidents
                port: 4000
                location_name: accidents_pipeline
'
```

### 3. Vérifier le déploiement

```bash
# Status de l'Application
argocd app get dagster-accidents-pipeline

# Pods dans le namespace dagster
kubectl get pods -n dagster

# Logs du user-code
kubectl logs -n dagster -l app=dagster-user-code,component=accidents --tail=50
```

### 4. Accéder à l'UI Dagster

Accédez à https://dagster.tgu.ovh et vérifiez que :
- Le workspace `accidents_pipeline` apparaît
- Les 4 assets bronze/silver/gold sont visibles
- Le graphe de dépendances est correct

## Workflow de mise à jour

### Modification du code

```bash
# 1. Commit changements
git add pipeline/ dagster_accidents/
git commit -m "feat: nouvelle feature"
git push

# 2. Rebuild et push l'image
docker build -f Dockerfile.dagster -t localhost:32000/accidents-dagster:latest .
docker push localhost:32000/accidents-dagster:latest

# 3. Redémarrer le pod (ArgoCD ne détecte pas les changements d'image)
kubectl rollout restart deployment/dagster-user-deployment-accidents -n dagster
```

### Modification de la configuration k8s

```bash
# 1. Modifier les manifests
nano k8s/dagster/deployment-dagster-user.yaml

# 2. Commit et push
git add k8s/dagster/
git commit -m "chore: update resources"
git push

# 3. ArgoCD sync automatiquement
argocd app sync dagster-accidents-pipeline
```

## Configuration

### Variables d'environnement DuckLake

Définies dans [configmap-dagster-ducklake.yaml](configmap-dagster-ducklake.yaml) :

| Variable | Valeur | Description |
|----------|--------|-------------|
| `DUCKLAKE_DATABASE_URL` | `postgres:dbname=data host=postgresql.datalab...` | Catalogue DuckLake |
| `DUCKLAKE_DATA_PATH` | `s3://accidents-bucket/ducklake/` | Stockage Parquet |
| `AWS_ENDPOINT_URL` | `https://rustfs.tgu.ovh` | Endpoint RustFS |
| `AWS_ACCESS_KEY_ID` | `rustfsadmin` (Secret) | Credentials RustFS |
| `AWS_SECRET_ACCESS_KEY` | `rustfsadmin` (Secret) | Credentials RustFS |

### Ressources allouées

Dans [deployment-dagster-user.yaml](deployment-dagster-user.yaml) :

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

## Troubleshooting

### L'application reste en "OutOfSync"

```bash
# Forcer le sync
argocd app sync dagster-accidents-pipeline --force

# Vérifier les différences
argocd app diff dagster-accidents-pipeline
```

### Le workspace n'apparaît pas dans Dagster

Vérifier que le webserver a bien la configuration :

```bash
# Redémarrer le webserver après modification
kubectl rollout restart deployment/dagster-webserver -n dagster

# Vérifier les logs
kubectl logs -n dagster deployment/dagster-webserver --tail=100 | grep workspace
```

### Erreur de connexion gRPC

Vérifier que le Service est accessible :

```bash
# Depuis le webserver
kubectl exec -n dagster deployment/dagster-webserver -- \
  curl -v dagster-user-code-accidents:4000

# Vérifier le Service existe
kubectl get svc -n dagster dagster-user-code-accidents
```

### Erreurs DuckLake

```bash
# Tester depuis le pod
kubectl exec -n dagster deployment/dagster-user-deployment-accidents -- \
  python -c "from pipeline.config import ensure_connection; conn = ensure_connection(); print('OK')"

# Vérifier connectivité Postgres
kubectl exec -n dagster deployment/dagster-user-deployment-accidents -- \
  nc -zv postgresql.datalab.svc.cluster.local 5432

# Vérifier connectivité RustFS
kubectl exec -n dagster deployment/dagster-user-deployment-accidents -- \
  curl -I https://rustfs.tgu.ovh
```

## Ressources créées

L'Application ArgoCD crée dans le namespace `dagster` :

- **ConfigMap** : `dagster-ducklake-config` (variables DuckLake)
- **Secret** : `dagster-ducklake-secrets` (credentials)
- **PVC** : `dagster-models-pvc` (5Gi pour artefacts ML)
- **Deployment** : `dagster-user-deployment-accidents` (user-code)
- **Service** : `dagster-user-code-accidents` (ClusterIP:4000)

