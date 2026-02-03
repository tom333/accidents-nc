# Configuration Helm Dagster pour accidents-pipeline

## Problème

Le `K8sRunLauncher` configuré dans le Helm chart Dagster principal a besoin de connaître l'image Docker à utiliser pour lancer les runs. Sans cette configuration, il essaie d'ajouter un tag `dagster/image` avec une valeur None, ce qui provoque une erreur.

## Solution

Il faut modifier l'Application ArgoCD qui déploie le Helm chart Dagster principal pour ajouter la configuration du workspace avec l'image du user-code.

### 1. Récupérer l'Application ArgoCD actuelle

```bash
kubectl get application dagster -n argocd -o yaml > dagster-app-backup.yaml
```

### 2. Modifier la configuration Helm

Éditer l'Application ArgoCD du Helm chart Dagster :

```bash
kubectl edit application dagster -n argocd
```

Ajouter/modifier la section `spec.source.helm.values` :

```yaml
spec:
  source:
    helm:
      values: |
        # ... configuration existante ...
        
        dagsterWebserver:
          workspace:
            enabled: true
            servers:
              - host: dagster-user-code-accidents
                port: 4000
                location_name: accidents_pipeline
        
        runLauncher:
          type: K8sRunLauncher
          config:
            k8sRunLauncher:
              # Image par défaut pour les runs (peut être overridée par le user-code)
              image:
                repository: localhost:32000/accidents-dagster
                tag: latest
                pullPolicy: Always
              # Namespace où lancer les jobs
              jobNamespace: dagster
              # Réutiliser les secrets/configmaps du user-code
              envConfigMaps:
                - name: dagster-ducklake-config
              envSecrets:
                - name: rustfs-credentials-dagster
```

### 3. Alternative : Configuration via annotations du deployment

Si vous ne pouvez pas modifier le Helm chart principal, ajoutez ces annotations au deployment user-code :

```yaml
metadata:
  annotations:
    dagster/image: "localhost:32000/accidents-dagster:latest"
```

Cette approche permet au launcher K8s de détecter automatiquement l'image à utiliser.

### 4. Vérifier la configuration

Après modification :

```bash
# Attendre la synchronisation ArgoCD
argocd app sync dagster

# Vérifier les pods
kubectl get pods -n dagster

# Tester la matérialisation d'un asset
# Dans l'UI Dagster : https://dagster.tgu.ovh
```

## Différence executor vs runLauncher

- **runLauncher** : Lance le job entier dans un pod K8s (configuré dans le Helm chart)
- **executor** : Définit comment exécuter les steps du job (in_process, multiprocess, k8s_job_executor)

Pour notre cas, on utilise :
- `K8sRunLauncher` : pour lancer chaque run dans un pod K8s dédié
- Executor par défaut (`in_process`) : tous les steps s'exécutent dans le même pod du run

Si on voulait chaque step dans un pod séparé, on utiliserait `k8s_job_executor` en plus.
