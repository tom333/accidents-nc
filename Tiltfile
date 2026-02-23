# Tiltfile pour le projet Accidents NC
# Documentation: https://docs.tilt.dev/

# Configuration
config.define_string_list("to-edit", args=True)
cfg = config.parse()

# Contexte Kubernetes
allow_k8s_contexts('microk8s')

# Build de l'image Docker Dagster
docker_build(
    'localhost:32000/accidents-dagster',
    context='.',
    dockerfile='./Dockerfile.dagster',
    # Build args pour invalider le cache si nécessaire
    build_args={'CACHEBUST': '20260219v1'},
    # Live update: sync Python files sans rebuild complet
    live_update=[
        sync('./src/accidents', '/app/src/accidents'),
        sync('./dagster_pipeline', '/app/dagster_pipeline'),
        # Redémarrer le processus après sync
        run('kill -HUP 1', trigger=['./src/accidents', './dagster_pipeline']),
    ],
    # Ignore les fichiers volumineux
    ignore=[
        './data/',
        './routes.nc',
        './routes_with_features.pkl',
        './models/',
        './catboost_info/',
        './archive/',
        './exploration/',
        './.venv/',
    ]
)

# Déploiement Kubernetes
k8s_yaml([
    'k8s/dagster/deployment-dagster-user.yaml',
])

# Resource Dagster user-code avec port-forwarding
k8s_resource(
    'dagster-user-deployment-accidents',
    port_forwards='4000:4000',
    labels=['dagster'],
)

# Logs colorés
print("""
╔═══════════════════════════════════════════════════════════╗
║                  🚀 Tilt - Accidents NC                  ║
╚═══════════════════════════════════════════════════════════╝

📦 Image: localhost:32000/accidents-dagster
🔄 Live update: src/accidents/ et dagster_pipeline/
🌐 Port-forward: localhost:4000 (Dagster gRPC)

🔧 Commandes utiles:
  - Rebuild complet: Tilt UI > restart resource
  - Logs: Tilt UI ou `tilt logs dagster-user-deployment-accidents`
  - Arrêter: Ctrl+C puis `tilt down`

💡 Ouvrir Tilt UI: http://localhost:10350
""")
