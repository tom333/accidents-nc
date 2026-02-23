#!/usr/bin/env bash
set -euo pipefail

# ============================================
# Script de déploiement unifié - Accidents NC
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonctions utilitaires
log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; exit 1; }

# Vérifier les prérequis
check_prerequisites() {
    log_info "Vérification des prérequis..."
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl n'est pas installé"
    fi
    
    if ! command -v docker &> /dev/null; then
        log_error "docker n'est pas installé"
    fi
    
    log_success "Prérequis OK"
}

# Usage
usage() {
    cat <<EOF
Usage: $0 [COMPONENT] [OPTIONS]

Déploie les composants du projet Accidents NC sur Kubernetes.

COMPONENTS:
  dagster     Déployer Dagster user-code deployment
  streamlit   Déployer l'application Streamlit
  all         Déployer tous les composants (défaut)

OPTIONS:
  --build     Rebuild les images Docker avant déploiement
  --no-push   Ne pas push les images au registry
  -h, --help  Afficher cette aide

EXEMPLES:
  $0 all --build              # Build et déploie tout
  $0 dagster                  # Déploie uniquement Dagster
  $0 streamlit --build        # Build et déploie Streamlit

NOTES:
  - Nécessite un contexte K8s configuré (microk8s)
  - Les secrets doivent être créés manuellement (rustfs-credentials-*)
  - Registry par défaut: localhost:32000
EOF
}

# Build image Docker
build_image() {
    local component=$1
    local dockerfile=$2
    local image_name=$3
    
    log_info "Build de l'image $component..."
    
    cd "$PROJECT_ROOT"
    docker build \
        -f "$dockerfile" \
        -t "$image_name" \
        --build-arg CACHEBUST="$(date +%s)" \
        .
    
    log_success "Image $component buildée: $image_name"
}

# Push image au registry
push_image() {
    local image_name=$1
    
    log_info "Push de l'image $image_name..."
    docker push "$image_name"
    log_success "Image pushée au registry"
}

# Déployer Dagster
deploy_dagster() {
    log_info "Déploiement Dagster user-code..."
    
    kubectl apply -f "$PROJECT_ROOT/k8s/dagster/configmap-dagster-ducklake.yaml"
    kubectl apply -f "$PROJECT_ROOT/k8s/dagster/deployment-dagster-user.yaml"
    
    log_success "Dagster déployé"
    log_info "Port-forward: kubectl port-forward -n dagster deployment/dagster-user-deployment-accidents 4000:4000"
}

# Déployer Streamlit
deploy_streamlit() {
    log_info "Déploiement Streamlit..."
    
    # Créer namespace si inexistant
    kubectl create namespace accidents --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl apply -f "$PROJECT_ROOT/k8s/streamlit/configmap-streamlit-ducklake.yaml"
    kubectl apply -f "$PROJECT_ROOT/k8s/streamlit/deployment.yaml"
    kubectl apply -f "$PROJECT_ROOT/k8s/streamlit/service.yaml"
    
    log_success "Streamlit déployé"
    log_info "Port-forward: kubectl port-forward -n accidents svc/streamlit-service 8501:80"
}

# Variables
COMPONENT="all"
BUILD=false
PUSH=true

# Parser les arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        dagster|streamlit|all)
            COMPONENT=$1
            shift
            ;;
        --build)
            BUILD=true
            shift
            ;;
        --no-push)
            PUSH=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Argument inconnu: $1\nUtiliser --help pour l'aide"
            ;;
    esac
done

# Main
main() {
    echo ""
    log_info "🚀 Déploiement Accidents NC"
    echo ""
    
    check_prerequisites
    
    # Build si demandé
    if [ "$BUILD" = true ]; then
        if [ "$COMPONENT" = "dagster" ] || [ "$COMPONENT" = "all" ]; then
            build_image "Dagster" "./Dockerfile.dagster" "localhost:32000/accidents-dagster:latest"
            if [ "$PUSH" = true ]; then
                push_image "localhost:32000/accidents-dagster:latest"
            fi
        fi
        
        if [ "$COMPONENT" = "streamlit" ] || [ "$COMPONENT" = "all" ]; then
            build_image "Streamlit" "./Dockerfile.streamlit" "localhost:32000/accidents-streamlit:latest"
            if [ "$PUSH" = true ]; then
                push_image "localhost:32000/accidents-streamlit:latest"
            fi
        fi
    fi
    
    # Déploiement
    if [ "$COMPONENT" = "dagster" ] || [ "$COMPONENT" = "all" ]; then
        deploy_dagster
    fi
    
    if [ "$COMPONENT" = "streamlit" ] || [ "$COMPONENT" = "all" ]; then
        deploy_streamlit
    fi
    
    echo ""
    log_success "🎉 Déploiement terminé !"
    echo ""
    
    # Afficher l'état
    if [ "$COMPONENT" = "dagster" ] || [ "$COMPONENT" = "all" ]; then
        log_info "Dagster pods:"
        kubectl get pods -n dagster -l component=accidents
    fi
    
    if [ "$COMPONENT" = "streamlit" ] || [ "$COMPONENT" = "all" ]; then
        log_info "Streamlit pods:"
        kubectl get pods -n accidents -l app=streamlit
    fi
    
    echo ""
}

main
