#!/bin/bash
set -e

echo "🚀 Déploiement de l'application Accidents sur MicroK8s"
echo "=================================================="

# Variables
IMAGE_NAME="accidents-app"
REGISTRY="localhost:32000"
TAG="latest"
NAMESPACE="accidents"

# Couleurs pour les messages
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "\n${BLUE}📋 Vérification des prérequis...${NC}"

# Vérifier que microk8s est installé
if ! command -v microk8s &> /dev/null; then
    echo -e "${RED}❌ MicroK8s n'est pas installé${NC}"
    exit 1
fi

# Vérifier que les addons nécessaires sont activés
echo -e "${BLUE}🔌 Vérification des addons MicroK8s...${NC}"
microk8s status --wait-ready

# REQUIRED_ADDONS=("registry" "ingress" "storage")
# for addon in "${REQUIRED_ADDONS[@]}"; do
#     if ! microk8s status | grep -q "$addon: enabled"; then
#         echo -e "${RED}❌ L'addon $addon n'est pas activé${NC}"
#         echo "Activez-le avec: microk8s enable $addon"
#         exit 1
#     fi
# done

echo -e "${GREEN}✅ Tous les addons sont activés${NC}"

# Vérifier que les fichiers modèles existent
echo -e "\n${BLUE}📦 Vérification des fichiers modèles...${NC}"
REQUIRED_FILES=("accident_model.pkl" "atm_encoder.pkl" "features.pkl" "routes.nc")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}❌ Fichier manquant: $file${NC}"
        echo "Exécutez d'abord: marimo run accident_fetch_data.py"
        exit 1
    fi
done
echo -e "${GREEN}✅ Tous les fichiers modèles sont présents${NC}"

# Construire l'image Docker
echo -e "\n${BLUE}🔨 Construction de l'image Docker...${NC}"
docker build -t ${IMAGE_NAME}:${TAG} .

# Tagger pour la registry locale
echo -e "${BLUE}🏷️  Tagging de l'image pour la registry locale...${NC}"
docker tag ${IMAGE_NAME}:${TAG} ${REGISTRY}/${IMAGE_NAME}:${TAG}

# Pousser vers la registry locale
echo -e "${BLUE}📤 Push vers la registry locale...${NC}"
docker push ${REGISTRY}/${IMAGE_NAME}:${TAG}
echo -e "${GREEN}✅ Image poussée vers ${REGISTRY}/${IMAGE_NAME}:${TAG}${NC}"

# Créer le namespace
echo -e "\n${BLUE}🏗️  Création du namespace...${NC}"
microk8s kubectl apply -f k8s/namespace.yaml

# Déployer l'application
echo -e "\n${BLUE}🚀 Déploiement de l'application...${NC}"
microk8s kubectl apply -f k8s/deployment.yaml
microk8s kubectl apply -f k8s/service.yaml
microk8s kubectl apply -f k8s/ingress.yaml

# Attendre que le déploiement soit prêt
echo -e "${BLUE}⏳ Attente du déploiement...${NC}"
microk8s kubectl wait --for=condition=available --timeout=300s deployment/accidents-app -n ${NAMESPACE}

echo -e "\n${GREEN}✅ Déploiement terminé avec succès!${NC}"
echo -e "\n${BLUE}📊 Statut du déploiement:${NC}"
microk8s kubectl get all -n ${NAMESPACE}

echo -e "\n${BLUE}🌐 Accès à l'application:${NC}"
echo -e "  • Local: http://localhost (si ingress configuré)"
echo -e "  • Public: https://meteo-des-accidents.tgu.ovh"
echo ""
echo -e "${BLUE}💡 Commandes utiles:${NC}"
echo "  • Voir les logs: microk8s kubectl logs -f deployment/accidents-app -n ${NAMESPACE}"
echo "  • Voir les pods: microk8s kubectl get pods -n ${NAMESPACE}"
echo "  • Voir l'ingress: microk8s kubectl get ingress -n ${NAMESPACE}"
echo "  • Redémarrer: microk8s kubectl rollout restart deployment/accidents-app -n ${NAMESPACE}"
echo "  • Supprimer: microk8s kubectl delete namespace ${NAMESPACE}"
