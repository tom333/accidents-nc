# Configuration Tilt pour le projet Accidents NC

## Installation de Tilt

### Sur Linux (recommandé)

```bash
curl -fsSL https://raw.githubusercontent.com/tilt-dev/tilt/master/scripts/install.sh | bash
```

### Vérifier l'installation

```bash
tilt version
```

## Utilisation

### Démarrer le développement

```bash
# À la racine du projet
tilt up
```

Cela va :
1. Builder l'image Docker `localhost:32000/accidents-dagster`
2. Déployer le pod Kubernetes `dagster-user-deployment-accidents`
3. Ouvrir l'interface web Tilt sur http://localhost:10350
4. Port-forward automatique sur localhost:4000 (Dagster gRPC)

### Live Update

Tilt détecte automatiquement les changements dans :
- `pipeline/*.py`
- `dagster_accidents/*.py`

Quand tu modifies un fichier Python, Tilt :
1. **Sync** le fichier dans le conteneur (sans rebuild)
2. **Redémarre** le processus Dagster automatiquement
3. **Rafraîchit** les logs dans l'UI

⚡ **Temps de mise à jour : ~2-5 secondes** (au lieu de ~2 minutes avec rebuild)

### Commandes utiles

```bash
# Logs en temps réel
tilt logs dagster-user-deployment-accidents

# Rebuild complet forcé
tilt trigger dagster-user-deployment-accidents

# Arrêter proprement
tilt down

# Nettoyer toutes les ressources K8s créées
tilt down --delete-namespaces
```

### Interface Web Tilt

Ouvre automatiquement http://localhost:10350

Fonctionnalités :
- ✅ État de tous les services (vert = OK, rouge = erreur)
- 📊 Logs agrégés et filtrables
- 🔄 Historique des builds
- 🔨 Boutons pour forcer un rebuild
- 📈 Graphe de dépendances

## Avantages vs `./rebuild-and-deploy.sh`

| Caractéristique | rebuild-and-deploy.sh | Tilt |
|-----------------|----------------------|------|
| Rebuild auto sur changement | ❌ Manuel | ✅ Automatique |
| Temps de mise à jour | ~2 minutes (full rebuild) | ~2-5 secondes (sync) |
| Interface graphique | ❌ | ✅ UI web |
| Logs centralisés | ❌ kubectl logs | ✅ UI web + agrégation |
| Port-forwarding | ❌ Manuel | ✅ Automatique |
| État des services | ❌ kubectl get pods | ✅ Dashboard temps réel |

## Configuration avancée

### Désactiver live update

Si tu veux forcer un rebuild complet à chaque fois :

```python
# Dans Tiltfile, commenter le bloc live_update
docker_build(
    'localhost:32000/accidents-dagster',
    context='.',
    dockerfile='./Dockerfile.dagster',
    # live_update=[...],  # Commenter cette section
)
```

### Ajouter d'autres services

Pour monitorer d'autres pods Kubernetes :

```python
# Dans Tiltfile
k8s_yaml([
    'k8s/dagster/deployment-dagster-user.yaml',
    'k8s/autre-service.yaml',  # Ajouter ici
])

k8s_resource(
    'mon-autre-service',
    port_forwards='8080:8080',
    labels=['backend'],
)
```

### Variables d'environnement

```python
# Dans Tiltfile
docker_build(
    'localhost:32000/accidents-dagster',
    build_args={'BUILD_ENV': 'dev'},
    ...
)
```

## Troubleshooting

### Tilt ne détecte pas les changements

```bash
# Vérifier que les fichiers ne sont pas dans .dockerignore
cat .dockerignore

# Redémarrer Tilt
tilt down && tilt up
```

### Erreur "context not allowed"

```bash
# Vérifier le contexte Kubernetes
kubectl config current-context

# Si différent de 'microk8s', mettre à jour Tiltfile:
allow_k8s_contexts('votre-contexte')
```

### Port déjà utilisé

```bash
# Si port 10350 ou 4000 déjà utilisé
tilt up -- --port=10351  # UI Tilt sur port alternatif
```

## Ressources

- [Documentation Tilt](https://docs.tilt.dev/)
- [Exemples Tiltfile](https://github.com/tilt-dev/tilt-example-python)
- [Live Update Guide](https://docs.tilt.dev/live_update_tutorial.html)
