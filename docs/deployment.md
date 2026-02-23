# 🚢 Guide de Déploiement - Accidents NC

Ce guide détaille le déploiement de la plateforme Accidents NC sur Kubernetes avec ArgoCD.

---

## 📋 Table des Matières

1. [Prérequis](#prérequis)
2. [Architecture de Déploiement](#architecture-de-déploiement)
3. [Setup Kubernetes (microk8s)](#setup-kubernetes-microk8s)
4. [Configuration DuckLake](#configuration-ducklake)
5. [Déploiement Dagster](#déploiement-dagster)
6. [Déploiement Streamlit](#déploiement-streamlit)
7. [Configuration ArgoCD](#configuration-argocd)
8. [CI/CD Pipeline](#cicd-pipeline)
9. [Monitoring et Logs](#monitoring-et-logs)
10. [Troubleshooting Production](#troubleshooting-production)

---

## ✅ Prérequis

### Infrastructure

- **Serveur dédié** : 4 CPU, 16 GB RAM minimum
- **OS** : Ubuntu 22.04 LTS ou similaire
- **Stockage** : 100 GB SSD (pour images Docker + PVC)
- **Réseau** : IP publique + DNS configuré

### Outils

```bash
# Kubernetes
microk8s (1.28+)

# CLI Tools
kubectl
helm (3.12+)
argocd CLI

# CI/CD
GitHub Actions (pas d'installation requise)

# Container Registry
Registry local microk8s (localhost:32000)
```

### Domaines DNS

Configurer les enregistrements DNS :
```
dagster.tgu.ovh     → A record → [IP serveur]
streamlit.tgu.ovh   → A record → [IP serveur]
rustfs.tgu.ovh      → A record → [IP serveur]
```

---

## 🏗️ Architecture de Déploiement

### Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────┐
│                      INTERNET                                │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    INGRESS (microk8s)                        │
├──────────────────────┬──────────────────┬───────────────────┤
│  dagster.tgu.ovh     │  streamlit.      │  rustfs.tgu.ovh   │
│  :443 (HTTPS)        │  tgu.ovh:443     │  :443             │
└──────────────────────┴──────────────────┴───────────────────┘
           ↓                    ↓                    ↓
┌──────────────────────────────────────────────────────────────┐
│              KUBERNETES CLUSTER (microk8s)                    │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  NAMESPACE: dagster                                           │
│  ├── dagster-webserver (Deployment)                          │
│  ├── dagster-daemon (Deployment)                             │
│  ├── dagster-user-deployment (Deployment)                    │
│  └── dagster-postgresql (StatefulSet - optionnel)            │
│                                                               │
│  NAMESPACE: streamlit                                         │
│  └── streamlit-app (Deployment)                              │
│                                                               │
│  NAMESPACE: datalab                                           │
│  └── postgresql (StatefulSet) - DuckLake catalog             │
│                                                               │
│  NAMESPACE: ia-lab                                            │
│  └── rustfs-svc (Deployment) - S3 storage                    │
│                                                               │
│  NAMESPACE: argocd                                            │
│  ├── argocd-server (Deployment)                              │
│  ├── argocd-repo-server (Deployment)                         │
│  └── argocd-application-controller (StatefulSet)             │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Namespaces

| Namespace | Rôle | Services |
|-----------|------|----------|
| `dagster` | Orchestration data pipeline | webserver, daemon, user-code |
| `streamlit` | Application prédictions | streamlit-app |
| `datalab` | DuckLake catalog | PostgreSQL |
| `ia-lab` | DuckLake storage | RustFS S3 |
| `argocd` | GitOps CD | ArgoCD |

---

## ☸️ Setup Kubernetes (microk8s)

### Installation microk8s

```bash
# Installer microk8s
sudo snap install microk8s --classic --channel=1.28/stable

# Ajouter utilisateur au groupe
sudo usermod -a -G microk8s $USER
sudo chown -R $USER ~/.kube
newgrp microk8s

# Vérifier installation
microk8s status --wait-ready

# Alias kubectl
alias kubectl='microk8s kubectl'
echo "alias kubectl='microk8s kubectl'" >> ~/.bashrc
```

### Activer Extensions

```bash
# DNS (requis)
microk8s enable dns

# Storage (requis)
microk8s enable hostpath-storage

# Ingress (requis pour HTTPS)
microk8s enable ingress

# Registry local (requis pour images Docker)
microk8s enable registry

# MetalLB (LoadBalancer, optionnel)
microk8s enable metallb:10.64.140.43-10.64.140.49

# Vérifier extensions
microk8s status
```

### Configuration kubectl

```bash
# Exporter kubeconfig
microk8s config > ~/.kube/config

# Vérifier accès
kubectl get nodes
kubectl get namespaces
```

---

## 🦆 Configuration DuckLake

### Déployer PostgreSQL Catalog

```bash
# Créer namespace
kubectl create namespace datalab

# Créer secret credentials
kubectl create secret generic postgres-secrets \
  --from-literal=password=YOUR_POSTGRES_PASSWORD \
  -n datalab

# Déployer PostgreSQL
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgresql
  namespace: datalab
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgresql
  template:
    metadata:
      labels:
        app: postgresql
    spec:
      containers:
      - name: postgres
        image: postgres:15
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: accidents
        - name: POSTGRES_USER
          value: accidents
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secrets
              key: password
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: datalab
spec:
  selector:
    app: postgresql
  ports:
  - port: 5432
    targetPort: 5432
  clusterIP: None
EOF

# Vérifier déploiement
kubectl get pods -n datalab
kubectl logs -n datalab statefulset/postgresql
```

### Déployer RustFS S3

```bash
# Créer namespace
kubectl create namespace ia-lab

# Créer secret S3
kubectl create secret generic rustfs-secrets \
  --from-literal=access-key=YOUR_ACCESS_KEY \
  --from-literal=secret-key=YOUR_SECRET_KEY \
  -n ia-lab

# Déployer RustFS (ou MinIO)
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rustfs
  namespace: ia-lab
spec:
  replicas: 1
  selector:
    matchLabels:
      app: rustfs
  template:
    metadata:
      labels:
        app: rustfs
    spec:
      containers:
      - name: rustfs
        image: rustfs/rustfs:latest  # Ou minio/minio:latest
        ports:
        - containerPort: 9000
        - containerPort: 9001
        env:
        - name: RUSTFS_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: rustfs-secrets
              key: access-key
        - name: RUSTFS_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: rustfs-secrets
              key: secret-key
        volumeMounts:
        - name: rustfs-data
          mountPath: /data
      volumes:
      - name: rustfs-data
        persistentVolumeClaim:
          claimName: rustfs-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: rustfs-pvc
  namespace: ia-lab
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
---
apiVersion: v1
kind: Service
metadata:
  name: rustfs-svc
  namespace: ia-lab
spec:
  selector:
    app: rustfs
  ports:
  - name: api
    port: 9000
    targetPort: 9000
  - name: console
    port: 9001
    targetPort: 9001
EOF

# Vérifier déploiement
kubectl get pods -n ia-lab
kubectl get svc -n ia-lab
```

### Créer Bucket S3

```bash
# Port-forward vers RustFS
kubectl port-forward -n ia-lab svc/rustfs-svc 9000:9000 &

# Installer mc (MinIO Client)
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/

# Configurer alias
mc alias set local http://localhost:9000 YOUR_ACCESS_KEY YOUR_SECRET_KEY

# Créer bucket
mc mb local/accidents

# Vérifier
mc ls local/
```

---

## 🎭 Déploiement Dagster

### Option A : Helm Chart (recommandé)

```bash
# Créer namespace
kubectl create namespace dagster

# Créer secrets
kubectl create secret generic dagster-postgresql-secret \
  --from-literal=postgresql-password=YOUR_DAGSTER_PASSWORD \
  -n dagster

kubectl create secret generic ducklake-secrets \
  --from-literal=POSTGRES_URL=postgresql://accidents:***@postgres.datalab.svc.cluster.local:5432/accidents \
  --from-literal=S3_ENDPOINT=https://rustfs.tgu.ovh \
  --from-literal=S3_BUCKET=accidents \
  --from-literal=AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY \
  --from-literal=AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY \
  -n dagster

# Ajouter repo Helm Dagster
helm repo add dagster https://dagster-io.github.io/helm
helm repo update

# Créer values.yaml
cat > values-dagster.yaml <<EOF
dagsterWebserver:
  replicas: 1
  image:
    repository: dagster/dagster-k8s
    tag: 1.8.0
  service:
    type: ClusterIP
    port: 3000

dagsterDaemon:
  replicas: 1

runLauncher:
  type: K8sRunLauncher
  config:
    k8sRunLauncher:
      imagePullPolicy: Always

postgresql:
  enabled: false  # Utiliser PostgreSQL externe

dagsterUserDeployments:
  deployments:
  - name: "user-code"
    image:
      repository: localhost:32000/accidents-dagster
      tag: latest
      pullPolicy: Always
    dagsterApiGrpcArgs:
    - "-f"
    - "/opt/dagster/app/dagster_pipeline/definitions.py"
    port: 3030
    envSecrets:
    - name: ducklake-secrets

ingress:
  enabled: true
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
  - host: dagster.tgu.ovh
    paths:
    - path: /
      pathType: Prefix
  tls:
  - secretName: dagster-tls
    hosts:
    - dagster.tgu.ovh
EOF

# Installer Dagster
helm install dagster dagster/dagster \
  -n dagster \
  -f values-dagster.yaml

# Vérifier déploiement
kubectl get pods -n dagster
kubectl get ingress -n dagster
```

### Option B : Manifests K8s Manuels

Utiliser les manifests dans [infra/k8s/dagster/](../infra/k8s/dagster/) :

```bash
cd infra/k8s/dagster

# Créer ConfigMap
kubectl apply -f configmap-dagster-ducklake.yaml

# Créer PVC pour modèles
kubectl apply -f pvc-dagster-models.yaml

# Déployer Dagster
kubectl apply -f deployment-dagster-user.yaml

# Vérifier
kubectl get pods -n dagster -w
```

### Build et Push Image Dagster

```bash
# Build image
docker build -f infra/docker/Dockerfile.dagster -t localhost:32000/accidents-dagster:latest .

# Push vers registry microk8s
docker push localhost:32000/accidents-dagster:latest

# Vérifier image
curl http://localhost:32000/v2/_catalog
```

### Accéder à Dagster UI

**Via port-forward** (temporaire) :
```bash
kubectl port-forward -n dagster svc/dagster-webserver 3000:3000

# Ouvrir http://localhost:3000
```

**Via Ingress** (production) :
```bash
# Accéder à https://dagster.tgu.ovh
```

---

## 📊 Déploiement Streamlit

### Build Image Streamlit

```bash
# Build
docker build -f infra/docker/Dockerfile.streamlit -t localhost:32000/accidents-streamlit:latest .

# Push
docker push localhost:32000/accidents-streamlit:latest
```

### Déployer sur K8s

```bash
# Créer namespace
kubectl create namespace streamlit

# Créer ConfigMap
kubectl create configmap streamlit-config \
  --from-literal=POSTGRES_URL=postgresql://accidents:***@postgres.datalab.svc.cluster.local:5432/accidents \
  --from-literal=S3_ENDPOINT=https://rustfs.tgu.ovh \
  --from-literal=S3_BUCKET=accidents \
  -n streamlit

# Créer secret
kubectl create secret generic streamlit-secrets \
  --from-literal=AWS_ACCESS_KEY_ID=YOUR_KEY \
  --from-literal=AWS_SECRET_ACCESS_KEY=YOUR_SECRET \
  -n streamlit

# Déployer
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: streamlit-app
  namespace: streamlit
spec:
  replicas: 1
  selector:
    matchLabels:
      app: streamlit
  template:
    metadata:
      labels:
        app: streamlit
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
        - name: S3_ENDPOINT
          valueFrom:
            configMapKeyRef:
              name: streamlit-config
              key: S3_ENDPOINT
        - name: S3_BUCKET
          valueFrom:
            configMapKeyRef:
              name: streamlit-config
              key: S3_BUCKET
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
---
apiVersion: v1
kind: Service
metadata:
  name: streamlit-svc
  namespace: streamlit
spec:
  selector:
    app: streamlit
  ports:
  - port: 8501
    targetPort: 8501
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: streamlit-ingress
  namespace: streamlit
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  rules:
  - host: streamlit.tgu.ovh
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: streamlit-svc
            port:
              number: 8501
  tls:
  - secretName: streamlit-tls
    hosts:
    - streamlit.tgu.ovh
EOF

# Vérifier
kubectl get pods -n streamlit
kubectl get ingress -n streamlit
```

---

## 🔄 Configuration ArgoCD

### Installer ArgoCD

```bash
# Créer namespace
kubectl create namespace argocd

# Installer ArgoCD
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Attendre que les pods soient ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=argocd-server -n argocd --timeout=300s

# Obtenir password admin initial
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
# Copier le password

# Port-forward (temporaire)
kubectl port-forward svc/argocd-server -n argocd 8080:443

# Ouvrir https://localhost:8080
# Login: admin / [password copié]
```

### Installer ArgoCD CLI

```bash
# Linux
curl -sSL -o argocd-linux-amd64 https://github.com/argoproj/argo-cd/releases/latest/download/argocd-linux-amd64
sudo install -m 555 argocd-linux-amd64 /usr/local/bin/argocd
rm argocd-linux-amd64

# Vérifier
argocd version
```

### Configurer ArgoCD avec GitHub

```bash
# Login ArgoCD CLI
argocd login localhost:8080 --username admin --password [password]

# Ajouter repo GitHub
argocd repo add https://github.com/tom333/accidents-nc.git \
  --username YOUR_GITHUB_USERNAME \
  --password YOUR_GITHUB_TOKEN

# Vérifier repos
argocd repo list
```

### Créer Application Dagster

```bash
# Appliquer manifest ArgoCD Application
kubectl apply -f infra/k8s/dagster/argocd-application.yaml

# Ou via CLI
argocd app create dagster-accidents-pipeline \
  --repo https://github.com/tom333/accidents-nc.git \
  --path infra/k8s/dagster \
  --dest-server https://kubernetes.default.svc \
  --dest-namespace dagster \
  --sync-policy automated \
  --auto-prune \
  --self-heal

# Vérifier app
argocd app list
argocd app get dagster-accidents-pipeline

# Sync manuellement (première fois)
argocd app sync dagster-accidents-pipeline
```

### Auto-Sync Configuration

[infra/k8s/dagster/argocd-application.yaml](../infra/k8s/dagster/argocd-application.yaml) :

```yaml
spec:
  syncPolicy:
    automated:
      prune: true       # Supprime ressources obsolètes
      selfHeal: true    # Répare divergences automatiquement
    syncOptions:
    - CreateNamespace=true
```

**Comportement** :
- ArgoCD poll Git toutes les **3 minutes**
- Si changement détecté → **sync automatique**
- Si drift détecté (cluster ≠ Git) → **self-heal**

---

## 🚀 CI/CD Pipeline

### GitHub Actions Setup

**Secrets à configurer dans GitHub** :

1. Aller sur GitHub repo : Settings → Secrets and variables → Actions
2. Ajouter secrets :

```
KUBE_CONFIG          # Contenu de ~/.kube/config (base64)
REGISTRY_URL         # localhost:32000 (ou autre)
DOCKER_USERNAME      # (optionnel si registry local)
DOCKER_PASSWORD      # (optionnel)
```

**Encoder kubeconfig** :
```bash
cat ~/.kube/config | base64 -w 0
# Copier output dans KUBE_CONFIG secret
```

### Workflows GitHub Actions

Déjà créés dans [.github/workflows/](../.github/workflows/) :

1. **ci-tests.yml** : Tests automatiques
2. **ci-build.yml** : Build images Docker
3. **cd-update-manifests.yml** : Update K8s manifests + ArgoCD

**Tester le pipeline** :

```bash
# 1. Commit changement
git add src/accidents/bronze/ingest.py
git commit -m "feat(bronze): improve error handling"
git push origin main

# 2. Vérifier GitHub Actions
# → https://github.com/tom333/accidents-nc/actions
# → CI Tests s'exécute (~5min)
# → CI Build s'exécute (~10min)
# → CD Update Manifests s'exécute (~1min)

# 3. Vérifier ArgoCD sync
kubectl get application -n argocd
argocd app get dagster-accidents-pipeline

# 4. Vérifier pods redémarrés
kubectl get pods -n dagster -w

# Total: ~22 minutes du push au déploiement
```

---

## 📊 Monitoring et Logs

### Logs Pods

```bash
# Logs Dagster webserver
kubectl logs -n dagster deployment/dagster-webserver -f

# Logs Dagster daemon
kubectl logs -n dagster deployment/dagster-daemon -f

# Logs user-code
kubectl logs -n dagster deployment/dagster-user-deployment -f

# Logs Streamlit
kubectl logs -n streamlit deployment/streamlit-app -f

# Logs PostgreSQL
kubectl logs -n datalab statefulset/postgresql -f

# Logs RustFS
kubectl logs -n ia-lab deployment/rustfs -f
```

### Events Kubernetes

```bash
# Events récents (tous namespaces)
kubectl get events --sort-by='.lastTimestamp' -A

# Events namespace dagster
kubectl get events -n dagster --sort-by='.lastTimestamp'

# Décrire pod (debug)
kubectl describe pod -n dagster dagster-user-deployment-xxx
```

### Métriques Pods

```bash
# CPU/Memory tous pods
kubectl top pods -A

# CPU/Memory namespace dagster
kubectl top pods -n dagster

# CPU/Memory nodes
kubectl top nodes
```

### ArgoCD UI

**Accéder à ArgoCD UI** :

```bash
# Port-forward
kubectl port-forward svc/argocd-server -n argocd 8080:443

# Ouvrir https://localhost:8080
# Voir status apps, history, events
```

### Dagster UI (Monitoring)

**Runs History** :
- Accéder à https://dagster.tgu.ovh
- Onglet "Runs"
- Voir succès/échecs, durée, logs

**Assets Status** :
- Onglet "Assets"
- Voir dernière matérialisation, dépendances

---

## 🛠️ Troubleshooting Production

### Pod CrashLoopBackOff

**Symptômes** :
```bash
kubectl get pods -n dagster
# dagster-user-deployment-xxx   0/1     CrashLoopBackOff
```

**Diagnostic** :
```bash
# Logs pod
kubectl logs -n dagster dagster-user-deployment-xxx

# Logs précédent container (si restart)
kubectl logs -n dagster dagster-user-deployment-xxx --previous

# Describe pod (voir events)
kubectl describe pod -n dagster dagster-user-deployment-xxx
```

**Solutions courantes** :
- Vérifier secrets existent : `kubectl get secrets -n dagster`
- Vérifier ConfigMap : `kubectl get configmap -n dagster`
- Vérifier image existe : `curl http://localhost:32000/v2/accidents-dagster/tags/list`

### ImagePullBackOff

**Symptômes** :
```
dagster-user-deployment-xxx   0/1     ImagePullBackOff
```

**Solutions** :
```bash
# Vérifier image dans registry
curl http://localhost:32000/v2/_catalog
curl http://localhost:32000/v2/accidents-dagster/tags/list

# Rebuild + push image
docker build -f infra/docker/Dockerfile.dagster -t localhost:32000/accidents-dagster:latest .
docker push localhost:32000/accidents-dagster:latest

# Forcer pull nouvelle image
kubectl rollout restart deployment/dagster-user-deployment -n dagster
```

### ArgoCD OutOfSync

**Symptômes** :
```bash
argocd app get dagster-accidents-pipeline
# Status: OutOfSync
```

**Solutions** :
```bash
# Sync manuellement
argocd app sync dagster-accidents-pipeline

# Forcer sync (ignore différences)
argocd app sync dagster-accidents-pipeline --force

# Hard refresh (delete + recreate)
argocd app sync dagster-accidents-pipeline --prune --force
```

### DuckLake Connection Errors

**Symptômes** :
```
psycopg2.OperationalError: could not connect to server
```

**Vérifications** :
```bash
# 1. PostgreSQL running?
kubectl get pods -n datalab

# 2. Service existe?
kubectl get svc -n datalab
# Devrait avoir: postgres.datalab.svc.cluster.local:5432

# 3. Tester connexion depuis pod
kubectl run -it --rm debug --image=postgres:15 --restart=Never -- \
  psql -h postgres.datalab.svc.cluster.local -U accidents -d accidents

# 4. Vérifier secrets
kubectl get secret ducklake-secrets -n dagster -o yaml
```

### S3 Access Errors

**Symptômes** :
```
botocore.exceptions.NoCredentialsError
```

**Vérifications** :
```bash
# 1. RustFS running?
kubectl get pods -n ia-lab

# 2. Port-forward test
kubectl port-forward -n ia-lab svc/rustfs-svc 9000:9000 &
mc ls local/

# 3. Vérifier secrets
kubectl get secret rustfs-secrets -n ia-lab -o yaml
kubectl get secret ducklake-secrets -n dagster -o yaml
```

### Ingress Not Working

**Symptômes** :
```
https://dagster.tgu.ovh → Connection refused
```

**Vérifications** :
```bash
# 1. Ingress controller running?
kubectl get pods -n ingress

# 2. Ingress resources?
kubectl get ingress -A

# 3. Certificate issued?
kubectl get certificate -A

# 4. Describe ingress
kubectl describe ingress dagster-ingress -n dagster

# 5. Test via port-forward
kubectl port-forward -n dagster svc/dagster-webserver 3000:3000
# Si ça marche → problème Ingress, sinon → problème service
```

### Disk Full

**Symptômes** :
```
Error: cannot write to disk: no space left on device
```

**Solutions** :
```bash
# 1. Vérifier usage disque
df -h

# 2. Nettoyer images Docker inutilisées
docker system prune -a --volumes

# 3. Nettoyer logs anciens
sudo journalctl --vacuum-time=7d

# 4. Augmenter PVC (si possible)
kubectl edit pvc dagster-models-pvc -n dagster
# Changer spec.resources.requests.storage
```

---

## 🔐 Sécurité Production

### Secrets Management

**Ne JAMAIS commiter** :
- Passwords
- API keys
- Tokens

**Utiliser Kubernetes Secrets** :
```bash
kubectl create secret generic my-secret \
  --from-literal=key=value \
  -n namespace
```

**Ou Sealed Secrets** (recommandé) :
```bash
# Installer Sealed Secrets controller
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/controller.yaml

# Créer sealed secret
echo -n 'my-password' | kubectl create secret generic my-secret \
  --dry-run=client \
  --from-file=password=/dev/stdin \
  -o yaml | \
kubeseal -o yaml > sealed-secret.yaml

# Commit sealed-secret.yaml (sûr)
git add sealed-secret.yaml
```

### HTTPS/TLS

**Installer cert-manager** :
```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Créer issuer Let's Encrypt
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: public
EOF
```

**Ingress utilisera automatiquement TLS** :
```yaml
metadata:
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - secretName: dagster-tls
    hosts:
    - dagster.tgu.ovh
```

### Network Policies (optionnel)

Restreindre trafic réseau entre pods :

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: dagster-policy
  namespace: dagster
spec:
  podSelector:
    matchLabels:
      app: dagster
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: dagster
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: datalab  # Autoriser PostgreSQL
    - namespaceSelector:
        matchLabels:
          name: ia-lab   # Autoriser S3
```

---

## 📈 Scaling et Performance

### Horizontal Pod Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: dagster-webserver-hpa
  namespace: dagster
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: dagster-webserver
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Resource Limits

```yaml
resources:
  requests:
    cpu: "500m"
    memory: "1Gi"
  limits:
    cpu: "2000m"
    memory: "4Gi"
```

---

## 📚 Ressources

### Documentation
- [Dagster on Kubernetes](https://docs.dagster.io/deployment/guides/kubernetes)
- [ArgoCD Getting Started](https://argo-cd.readthedocs.io/en/stable/getting_started/)
- [microk8s Documentation](https://microk8s.io/docs)

### Support
- **Issues GitHub** : https://github.com/tom333/accidents-nc/issues
- **Slack** : (à créer)
