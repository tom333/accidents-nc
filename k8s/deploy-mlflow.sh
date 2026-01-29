#!/bin/bash
set -e

echo "🚀 Déploiement complet du système accidents-prediction (MLflow + API)"
echo ""

# 1. Configurer l'infrastructure
echo "📦 Étape 1: Configuration de l'infrastructure microk8s"
bash k8s/setup-microk8s.sh

# Attendre que les services soient prêts
echo "⏳ Attente du démarrage des services..."
microk8s kubectl wait --for=condition=ready pod -l app=mlflow -n accidents --timeout=300s
microk8s kubectl wait --for=condition=ready pod -l app=minio -n accidents --timeout=300s

# 2. Builder les images Docker
echo ""
echo "🐳 Étape 2: Build des images Docker"

echo "  📦 Build de l'image API..."
docker build -f Dockerfile.api -t localhost:32000/accidents-api:latest .
docker push localhost:32000/accidents-api:latest

echo "  📦 Build de l'image Training..."
docker build -f Dockerfile.training -t localhost:32000/accidents-training:latest .
docker push localhost:32000/accidents-training:latest

# 3. Déployer l'API
echo ""
echo "🚀 Étape 3: Déploiement de l'API"
microk8s kubectl apply -f k8s/deployment-api.yaml

echo "⏳ Attente du démarrage de l'API..."
microk8s kubectl wait --for=condition=ready pod -l app=accident-api -n accidents --timeout=300s

# 4. Créer le CronJob d'entraînement
echo ""
echo "⏰ Étape 4: Configuration du CronJob annuel"
microk8s kubectl apply -f k8s/cronjob-training.yaml

# 5. Lancer le premier entraînement manuellement
echo ""
echo "🎓 Étape 5: Premier entraînement (optionnel)"
read -p "Lancer le premier entraînement maintenant ? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "🏃 Lancement du job d'entraînement..."
    microk8s kubectl create job --from=cronjob/annual-training initial-training -n accidents
    
    echo "📊 Suivi des logs du job (Ctrl+C pour quitter):"
    microk8s kubectl logs -f job/initial-training -n accidents
fi

# 6. Afficher les informations de connexion
echo ""
echo "✅ DÉPLOIEMENT TERMINÉ"
echo ""
echo "📍 Accès aux services:"
echo ""
echo "   🌐 API Prédiction:"
echo "      URL: http://$(microk8s kubectl get svc accident-api -n accidents -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"
echo "      Docs: http://$(microk8s kubectl get svc accident-api -n accidents -o jsonpath='{.status.loadBalancer.ingress[0].ip}')/docs"
echo ""
echo "   📊 MLflow UI (port-forward):"
echo "      microk8s kubectl port-forward -n accidents svc/mlflow 5000:5000"
echo "      Puis: http://localhost:5000"
echo ""
echo "   💾 MinIO Console (port-forward):"
echo "      microk8s kubectl port-forward -n accidents svc/minio 9001:9001"
echo "      Puis: http://localhost:9001 (minioadmin/minioadmin)"
echo ""
echo "🔧 Commandes utiles:"
echo ""
echo "   # Déclencher un entraînement manuellement"
echo "   microk8s kubectl create job --from=cronjob/annual-training manual-training-\$(date +%Y%m%d) -n accidents"
echo ""
echo "   # Voir les logs de l'API"
echo "   microk8s kubectl logs -f deployment/accident-api -n accidents"
echo ""
echo "   # Voir l'historique des jobs"
echo "   microk8s kubectl get jobs -n accidents"
echo ""
echo "   # Recharger le modèle sans redémarrer"
echo "   curl -X POST http://API_IP/reload-model"
echo ""
