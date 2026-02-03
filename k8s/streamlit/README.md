# Déploiement Kubernetes - Streamlit App

Manifests Kubernetes pour l'application Streamlit de visualisation.

## Fichiers

- `deployment.yaml` : Deployment principal Streamlit
- `deployment-api.yaml` : API backend (optionnel)
- `service.yaml` : Service ClusterIP
- `ingress.yaml` : Ingress pour accès externe
- `pvc.yaml` : PersistentVolumeClaim pour données
- `deploy.sh` : Script de déploiement automatisé

## Déploiement

```bash
./k8s/streamlit/deploy.sh
```

Ou manuellement :

```bash
kubectl apply -f k8s/streamlit/pvc.yaml
kubectl apply -f k8s/streamlit/deployment.yaml
kubectl apply -f k8s/streamlit/service.yaml
kubectl apply -f k8s/streamlit/ingress.yaml
```

## Accès

Via ingress configuré ou port-forward :

```bash
kubectl port-forward svc/streamlit-service 8501:8501
```

Puis accédez à http://localhost:8501
