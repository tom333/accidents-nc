#!/bin/bash
set -e

echo "🚀 Configuration de microk8s pour accidents-prediction"

# 1. Activer les addons nécessaires
echo "📦 Activation des addons microk8s..."
microk8s enable dns storage ingress registry

# 2. Créer le namespace
echo "📁 Création du namespace accidents..."
microk8s kubectl create namespace accidents --dry-run=client -o yaml | microk8s kubectl apply -f -

# 3. Installer MinIO (stockage S3-compatible)
echo "💾 Installation de MinIO..."
microk8s kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: minio-pvc
  namespace: accidents
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: minio
  namespace: accidents
spec:
  replicas: 1
  selector:
    matchLabels:
      app: minio
  template:
    metadata:
      labels:
        app: minio
    spec:
      containers:
      - name: minio
        image: minio/minio:latest
        args:
        - server
        - /data
        - --console-address
        - ":9001"
        env:
        - name: MINIO_ROOT_USER
          value: "minioadmin"
        - name: MINIO_ROOT_PASSWORD
          value: "minioadmin"
        ports:
        - containerPort: 9000
        - containerPort: 9001
        volumeMounts:
        - name: storage
          mountPath: /data
      volumes:
      - name: storage
        persistentVolumeClaim:
          claimName: minio-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: minio
  namespace: accidents
spec:
  selector:
    app: minio
  ports:
  - name: api
    port: 9000
    targetPort: 9000
  - name: console
    port: 9001
    targetPort: 9001
EOF

# 4. Installer MLflow
echo "📊 Installation de MLflow..."
microk8s kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mlflow-pvc
  namespace: accidents
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  namespace: accidents
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
      - name: mlflow
        image: ghcr.io/mlflow/mlflow:v2.10.0
        command:
        - mlflow
        - server
        - --host
        - "0.0.0.0"
        - --port
        - "5000"
        - --backend-store-uri
        - "sqlite:///mlflow/mlflow.db"
        - --default-artifact-root
        - "s3://mlflow-artifacts/"
        env:
        - name: MLFLOW_S3_ENDPOINT_URL
          value: "http://minio:9000"
        - name: AWS_ACCESS_KEY_ID
          value: "minioadmin"
        - name: AWS_SECRET_ACCESS_KEY
          value: "minioadmin"
        ports:
        - containerPort: 5000
        volumeMounts:
        - name: mlflow-data
          mountPath: /mlflow
      volumes:
      - name: mlflow-data
        persistentVolumeClaim:
          claimName: mlflow-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow
  namespace: accidents
spec:
  selector:
    app: mlflow
  ports:
  - port: 5000
    targetPort: 5000
EOF

# 5. Créer le bucket MinIO pour MLflow
echo "🪣 Configuration du bucket MLflow..."
microk8s kubectl run minio-init --rm -i --restart=Never --image=minio/mc:latest --namespace=accidents -- sh -c "
  mc alias set myminio http://minio:9000 minioadmin minioadmin
  mc mb myminio/mlflow-artifacts --ignore-existing
  mc anonymous set download myminio/mlflow-artifacts
"

echo "✅ Infrastructure prête!"
echo ""
echo "📍 Accès aux services:"
echo "   - MLflow UI:   http://localhost:5000 (port-forward: microk8s kubectl port-forward -n accidents svc/mlflow 5000:5000)"
echo "   - MinIO Console: http://localhost:9001 (port-forward: microk8s kubectl port-forward -n accidents svc/minio 9001:9001)"
