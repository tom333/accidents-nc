# 🐳 Déploiement Docker - Application Accidents

Ce guide explique comment conteneuriser et déployer l'application de prédiction des accidents avec Docker.

## 📋 Prérequis

- Docker Engine 20.10+ et Docker Compose 2.0+
- Fichiers modèles générés (`accident_model.pkl`, `atm_encoder.pkl`, `features.pkl`, `routes.nc`)

## 🏗️ Architecture du Container

Le Dockerfile utilise :
- **Base**: Python 3.13-slim pour une image légère
- **Dépendances système**: GDAL, GEOS, PROJ pour les librairies géospatiales
- **Package manager**: uv pour des installations rapides
- **Sécurité**: Utilisateur non-root (`app_user`)
- **Monitoring**: Health check sur `/health`
- **Port**: 8080 (configurable)

## 🚀 Démarrage Rapide

### Option 1 : Docker Compose (recommandé)

```bash
# Construire et démarrer l'application
docker compose up -d

# Voir les logs
docker compose logs -f

# Arrêter
docker compose down
```

L'application sera accessible sur http://localhost:8080

### Option 2 : Docker CLI

```bash
# Construire l'image
docker build -t accidents-app .

# Lancer le container
docker run -d \
  -p 8080:8080 \
  --name accidents \
  --restart unless-stopped \
  accidents-app

# Voir les logs
docker logs -f accidents

# Arrêter et supprimer
docker stop accidents && docker rm accidents
```

## 📦 Contenu du Container

**Fichiers copiés** :
- `predict_map.py` : Application Marimo principale
- `accident_model.pkl` : Modèle LightGBM entraîné
- `atm_encoder.pkl` : Encodeur des conditions météo
- `features.pkl` : Liste des features du modèle
- `routes.nc` : Réseau routier OSM de Nouvelle-Calédonie

**Dépendances installées** (depuis `pyproject.toml`) :
```
marimo, geopandas, osmnx, folium, lightgbm, xgboost, 
catboost, scikit-learn, pandas, numpy, shapely
```

## 🔧 Configuration Avancée

### Variables d'environnement

Modifiez `docker-compose.yml` pour personnaliser :

```yaml
environment:
  - MARIMO_LOG_LEVEL=DEBUG  # INFO, WARNING, ERROR
  - MARIMO_ALLOW_ORIGINS=*  # CORS origins
```

### Changer le port

```yaml
ports:
  - "3000:8080"  # Accessible sur http://localhost:3000
```

### Volumes persistants

Pour éviter de reconstruire l'image à chaque changement de modèle :

```yaml
volumes:
  - ./accident_model.pkl:/app/accident_model.pkl:ro
  - ./atm_encoder.pkl:/app/atm_encoder.pkl:ro
  - ./features.pkl:/app/features.pkl:ro
  - ./routes.nc:/app/routes.nc:ro
```

Le flag `:ro` (read-only) empêche toute modification depuis le container.

## 🏥 Monitoring

### Health Check

Le container expose deux endpoints de monitoring :

```bash
# Vérifier la santé de l'application
curl http://localhost:8080/health

# Statut détaillé du serveur
curl http://localhost:8080/api/status
```

Docker Compose vérifie automatiquement la santé toutes les 30 secondes.

### Logs

```bash
# Logs en temps réel
docker compose logs -f

# Logs des dernières 100 lignes
docker compose logs --tail 100

# Logs avec timestamps
docker compose logs -t
```

## 🌐 Déploiement en Production

### 1. Variables sensibles

Utilisez un fichier `.env` pour les secrets :

```bash
# .env
MARIMO_AUTH_TOKEN=votre_token_secret
MARIMO_LOG_LEVEL=WARNING
```

Puis dans `docker-compose.yml` :

```yaml
env_file:
  - .env
```

### 2. Reverse Proxy (nginx)

Exemple de configuration nginx :

```nginx
server {
    listen 80;
    server_name accidents.example.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3. HTTPS avec Let's Encrypt

```bash
# Installer certbot
apt install certbot python3-certbot-nginx

# Obtenir un certificat
certbot --nginx -d accidents.example.com
```

### 4. Limites de ressources

Dans `docker-compose.yml` :

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '1'
      memory: 2G
```

## 🐛 Dépannage

### L'image ne se construit pas

**Problème** : Erreur lors de l'installation des dépendances géospatiales

```bash
# Reconstruire sans cache
docker compose build --no-cache
```

**Problème** : Espace disque insuffisant

```bash
# Nettoyer les images inutilisées
docker system prune -a
```

### Le container ne démarre pas

**Problème** : Fichiers modèles manquants

```bash
# Vérifier les fichiers requis
ls -lh *.pkl routes.nc

# Si manquants, lancer l'entraînement
marimo edit accident_fetch_data.py
```

**Problème** : Port déjà utilisé

```bash
# Trouver le processus utilisant le port 8080
lsof -i :8080

# Ou changer le port dans docker-compose.yml
```

### L'application est lente

**Problème** : Ressources limitées

```bash
# Voir l'utilisation des ressources
docker stats accidents-app

# Augmenter les limites dans docker-compose.yml
```

## 📊 Métriques et Performance

### Taille de l'image

```bash
docker images accidents-app
# Attendu : ~1.5-2GB (avec toutes les dépendances géospatiales)
```

### Temps de démarrage

- **Build initial** : 5-10 min (dépendances système + Python)
- **Rebuild** (code changé) : 10-30s (cache Docker)
- **Démarrage container** : 5-10s (chargement modèles)

### Optimisations

Pour réduire la taille de l'image :

```dockerfile
# Multi-stage build (avancé)
FROM python:3.13-slim AS builder
# ... installation dépendances ...

FROM python:3.13-slim
COPY --from=builder /usr/local /usr/local
```

## 🔄 Mise à Jour

### Nouveau modèle entraîné

```bash
# Option 1 : Volumes (pas de rebuild)
docker compose restart

# Option 2 : Rebuild complet
docker compose up -d --build
```

### Mise à jour de l'application

```bash
# Arrêter, reconstruire, redémarrer
docker compose down
docker compose up -d --build
```

## 📚 Ressources

- [Documentation Docker](https://docs.docker.com/)
- [Documentation Marimo Deployment](https://docs.marimo.io/guides/deploying/deploying_docker/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Health Checks](https://docs.docker.com/engine/reference/builder/#healthcheck)

## 🆘 Support

En cas de problème :

1. Vérifier les logs : `docker compose logs -f`
2. Tester le health check : `curl http://localhost:8080/health`
3. Vérifier les ressources : `docker stats`
4. Reconstruire proprement : `docker compose down && docker compose up -d --build`
