#!/bin/bash
set -e

echo "🚀 Déploiement Dagster user-deployment pour pipeline accidents"

# 1. Build Docker image
echo "📦 Build de l'image Docker..."
docker build -f Dockerfile.dagster -t localhost:32000/accidents-dagster:latest .

# 2. Push vers le registre interne microk8s
echo "🏷️  Push vers le registre microk8s (localhost:32000)..."
docker push localhost:32000/accidents-dagster:latest

# 3. Créer le namespace si nécessaire (optionnel)
# kubectl create namespace dagster --dry-run=client -o yaml | kubectl apply -f -

# 4. Appliquer les manifests k8s
echo "📝 Application des manifests k8s..."

# PVC pour les modèles
kubectl apply -f k8s/dagster/pvc-dagster-models.yaml

# ConfigMap et Secret pour DuckLake
echo "⚙️  Configuration DuckLake..."
kubectl apply -f k8s/dagster/configmap-dagster-ducklake.yaml

# Deployment et Service
echo "🎯 Déploiement du user-code..."
kubectl apply -f k8s/dagster/deployment-dagster-user.yaml

# 5. Attendre que le pod soit ready
echo "⏳ Attente du démarrage du pod..."
kubectl wait --for=condition=ready pod -l app=dagster-user-code,component=accidents --timeout=180s

# 6. Vérifier les logs
echo "📋 Logs du pod :"
kubectl logs -l app=dagster-user-code,component=accidents --tail=50

echo ""
echo "✅ Déploiement terminé !"
echo ""
echo "🔍 Prochaines étapes :"
echo "  1. Configurer le workspace Dagster avec k8s/workspace.yaml"
echo "  2. Redémarrer le webserver Dagster pour qu'il détecte le nouveau code"
echo "  3. Vérifier dans l'UI Dagster que le workspace 'accidents_pipeline' apparaît"
echo "  4. Lancer une matérialisation de gold_models"
echo ""
echo "📖 Commandes utiles :"
echo "  kubectl get pods -l app=dagster-user-code"
echo "  kubectl logs -f -l app=dagster-user-code,component=accidents"
echo "  kubectl describe pod -l app=dagster-user-code,component=accidents"
