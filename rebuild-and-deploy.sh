#!/bin/bash
set -e

echo "🔨 Building Docker image..."
DOCKER_BUILDKIT=1 docker build -f Dockerfile.dagster -t localhost:32000/accidents-dagster:latest .

echo ""
echo "📤 Pushing to registry..."
docker push localhost:32000/accidents-dagster:latest

echo ""
echo "🔄 Restarting deployment..."
kubectl rollout restart deployment dagster-user-deployment-accidents -n dagster

echo ""
echo "⏳ Waiting for pod to be ready..."
kubectl rollout status deployment dagster-user-deployment-accidents -n dagster --timeout=120s

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Pod status:"
kubectl get pods -n dagster | grep user-deployment

echo ""
echo "📝 Recent logs:"
kubectl logs -n dagster deployment/dagster-user-deployment-accidents --tail=10

echo ""
echo "💡 To follow logs: kubectl logs -f -n dagster deployment/dagster-user-deployment-accidents"
