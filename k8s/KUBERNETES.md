# 🚀 Déploiement Kubernetes sur MicroK8s

Ce guide explique comment déployer l'application de prédiction des accidents sur un cluster MicroK8s local.

## 📋 Prérequis

### Installation de MicroK8s

```bash
# Installer MicroK8s
sudo snap install microk8s --classic

# Ajouter l'utilisateur au groupe microk8s
sudo usermod -a -G microk8s $USER
sudo chown -f -R $USER ~/.kube
newgrp microk8s

# Vérifier le statut
microk8s status --wait-ready
```

### Activer les addons nécessaires

```bash
# DNS (résolution de noms interne)
microk8s enable dns

# Registry locale
microk8s enable registry

# Ingress (accès externe)
microk8s enable ingress

# Storage (volumes persistants)
microk8s enable storage

# Cert-manager (pour HTTPS avec Let's Encrypt)
microk8s enable cert-manager

# Vérifier que tout est activé
microk8s status
```

### Préparer les fichiers modèles

```bash
# Générer les modèles si pas déjà fait
marimo run accident_fetch_data.py
```

## 🏗️ Architecture Kubernetes

```
┌─────────────────────────────────────────────┐
│ Ingress (meteo-des-accidents.tgu.ovh)      │
│ - TLS/HTTPS automatique                     │
│ - Nginx Ingress Controller                  │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│ Service (ClusterIP)                         │
│ - Port 80 → 8080                            │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│ Deployment (accidents-app)                  │
│ - 1 replica (ajustable)                     │
│ - Image: localhost:32000/accidents-app      │
│ - Health checks: /health                    │
│ - Resources: 2Gi RAM, 500m CPU              │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│ PersistentVolumeClaim                       │
│ - Stockage des modèles ML                   │
│ - 2Gi, ReadOnlyMany                         │
└─────────────────────────────────────────────┘
```

## 🚀 Déploiement Rapide

### Option 1 : Script automatique (recommandé)

```bash
# Rendre le script exécutable
chmod +x k8s/deploy.sh

# Lancer le déploiement complet
./k8s/deploy.sh
```

Le script effectue automatiquement :
1. ✅ Vérification des prérequis
2. 🔨 Build de l'image Docker
3. 📤 Push vers la registry locale
4. 🏗️ Création du namespace et PVC
5. 📋 Copie des fichiers modèles
6. 🚀 Déploiement de l'application
7. 🌐 Configuration de l'ingress

### Option 2 : Déploiement manuel

#### 1. Build et push de l'image

```bash
# Build
docker build -t accidents-app:latest .

# Tag pour la registry locale
docker tag accidents-app:latest localhost:32000/accidents-app:latest

# Push
docker push localhost:32000/accidents-app:latest
```

#### 2. Déploiement des manifests

```bash
# Créer le namespace
microk8s kubectl apply -f k8s/namespace.yaml

# Créer le PVC
microk8s kubectl apply -f k8s/pvc.yaml

# Copier les modèles dans le PVC (voir section dédiée ci-dessous)

# Déployer l'application
microk8s kubectl apply -f k8s/deployment.yaml
microk8s kubectl apply -f k8s/service.yaml
microk8s kubectl apply -f k8s/ingress.yaml
```

#### 3. Copier les fichiers modèles dans le PVC

```bash
# Créer un pod temporaire
cat <<EOF | microk8s kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: model-loader
  namespace: accidents
spec:
  containers:
  - name: loader
    image: busybox
    command: ['sh', '-c', 'sleep 3600']
    volumeMounts:
    - name: models
      mountPath: /models
  volumes:
  - name: models
    persistentVolumeClaim:
      claimName: accidents-models-pvc
EOF

# Attendre que le pod soit prêt
microk8s kubectl wait --for=condition=ready pod/model-loader -n accidents

# Copier les fichiers
microk8s kubectl cp accident_model.pkl accidents/model-loader:/models/
microk8s kubectl cp atm_encoder.pkl accidents/model-loader:/models/
microk8s kubectl cp features.pkl accidents/model-loader:/models/
microk8s kubectl cp routes.nc accidents/model-loader:/models/

# Supprimer le pod
microk8s kubectl delete pod model-loader -n accidents
```

## 🔐 Configuration HTTPS avec Let's Encrypt

### 1. Installer cert-manager

```bash
microk8s enable cert-manager
```

### 2. Créer un ClusterIssuer

```bash
cat <<EOF | microk8s kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: votre-email@example.com  # ⚠️ Remplacer par votre email
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### 3. Vérifier le certificat

```bash
# Voir les certificats
microk8s kubectl get certificates -n accidents

# Voir le statut détaillé
microk8s kubectl describe certificate accidents-tls-cert -n accidents
```

Le certificat sera automatiquement généré et renouvelé par cert-manager.

## 🌐 Configuration DNS

Pour accéder à l'application via `https://meteo-des-accidents.tgu.ovh`, configurez votre DNS :

```
Type: A
Host: meteo-des-accidents
Value: <IP_PUBLIC_DU_SERVEUR>
TTL: 3600
```

**Test local (sans DNS public)** :

```bash
# Ajouter au /etc/hosts
echo "127.0.0.1 meteo-des-accidents.tgu.ovh" | sudo tee -a /etc/hosts

# Tester
curl -k https://meteo-des-accidents.tgu.ovh
```

## 📊 Monitoring et Gestion

### Voir les ressources

```bash
# Tous les objets du namespace
microk8s kubectl get all -n accidents

# Pods avec détails
microk8s kubectl get pods -n accidents -o wide

# Logs de l'application
microk8s kubectl logs -f deployment/accidents-app -n accidents

# Événements
microk8s kubectl get events -n accidents --sort-by='.lastTimestamp'
```

### Accéder à un pod

```bash
# Shell interactif
microk8s kubectl exec -it deployment/accidents-app -n accidents -- /bin/bash

# Commande unique
microk8s kubectl exec deployment/accidents-app -n accidents -- ls -la /app
```

### Scaler l'application

```bash
# Augmenter à 3 réplicas
microk8s kubectl scale deployment/accidents-app -n accidents --replicas=3

# Vérifier
microk8s kubectl get pods -n accidents
```

### Mettre à jour l'application

```bash
# Rebuild et push de la nouvelle image
docker build -t accidents-app:latest .
docker tag accidents-app:latest localhost:32000/accidents-app:latest
docker push localhost:32000/accidents-app:latest

# Forcer le redéploiement
microk8s kubectl rollout restart deployment/accidents-app -n accidents

# Suivre le rollout
microk8s kubectl rollout status deployment/accidents-app -n accidents
```

### Health checks

```bash
# Port-forward pour tester localement
microk8s kubectl port-forward -n accidents service/accidents-service 8080:80

# Tester le health check
curl http://localhost:8080/health

# Voir le statut des probes
microk8s kubectl describe pod -n accidents | grep -A 5 "Liveness\|Readiness"
```

## 🔧 Configuration Avancée

### Limites de ressources

Modifier `k8s/deployment.yaml` :

```yaml
resources:
  requests:
    memory: "4Gi"  # Ressources garanties
    cpu: "1000m"
  limits:
    memory: "8Gi"  # Maximum autorisé
    cpu: "4000m"
```

### Variables d'environnement

Ajouter dans `k8s/deployment.yaml` :

```yaml
env:
- name: MARIMO_LOG_LEVEL
  value: "DEBUG"
- name: MARIMO_ALLOW_ORIGINS
  value: "*"
```

### Utiliser un ConfigMap

```bash
# Créer un ConfigMap
cat <<EOF | microk8s kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: accidents-config
  namespace: accidents
data:
  LOG_LEVEL: "INFO"
  PREDICTION_THRESHOLD: "0.85"
EOF

# Utiliser dans le deployment
envFrom:
- configMapRef:
    name: accidents-config
```

## 🐛 Dépannage

### Pods en erreur

```bash
# Voir les logs
microk8s kubectl logs deployment/accidents-app -n accidents

# Logs des conteneurs précédents (après crash)
microk8s kubectl logs deployment/accidents-app -n accidents --previous

# Décrire le pod (voir events)
microk8s kubectl describe pod -n accidents <nom-du-pod>
```

### Image non trouvée

```bash
# Vérifier que l'image est dans la registry
curl http://localhost:32000/v2/_catalog

# Vérifier le tag
curl http://localhost:32000/v2/accidents-app/tags/list

# Re-push si nécessaire
docker push localhost:32000/accidents-app:latest
```

### Ingress ne fonctionne pas

```bash
# Vérifier l'ingress controller
microk8s kubectl get pods -n ingress

# Vérifier l'ingress
microk8s kubectl describe ingress accidents-ingress -n accidents

# Voir les logs de nginx
microk8s kubectl logs -n ingress -l app.kubernetes.io/name=ingress-nginx
```

### PVC non bound

```bash
# Voir le statut
microk8s kubectl get pvc -n accidents

# Voir les PV disponibles
microk8s kubectl get pv

# Forcer la création
microk8s kubectl delete pvc accidents-models-pvc -n accidents
microk8s kubectl apply -f k8s/pvc.yaml
```

## 🗑️ Nettoyage

### Supprimer l'application

```bash
# Supprimer tout le namespace (application + PVC)
microk8s kubectl delete namespace accidents

# Ou supprimer individuellement
microk8s kubectl delete -f k8s/
```

### Supprimer l'image de la registry

```bash
# L'image reste dans la registry locale, utiliser:
docker image rm localhost:32000/accidents-app:latest
```

## 📚 Ressources

- [Documentation MicroK8s](https://microk8s.io/docs)
- [Kubernetes Ingress](https://kubernetes.io/docs/concepts/services-networking/ingress/)
- [Cert-Manager](https://cert-manager.io/docs/)
- [Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)

## 🆘 Support

Commandes de diagnostic complètes :

```bash
# État du cluster
microk8s status
microk8s inspect

# Tous les logs
microk8s kubectl logs -n accidents --all-containers=true -l app=accidents

# Export de toutes les ressources
microk8s kubectl get all,pvc,ingress -n accidents -o yaml > accidents-state.yaml
```
