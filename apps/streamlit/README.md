# Application Streamlit - Prédiction Accidents NC

## Lancement

```bash
# Depuis la racine du projet
PYTHONPATH=$PWD streamlit run apps/streamlit/app.py

# Ou avec un port spécifique
PYTHONPATH=$PWD streamlit run apps/streamlit/app.py --server.port 8502
```

L'application sera accessible sur http://localhost:8501 (ou 8502)

## Architecture

L'application utilise :
- **DuckLake** : Chargement des données depuis `silver.full_dataset`
- **Streamlit** : Interface interactive
- **Folium** : Visualisation cartographique
- **NumPy/Pandas** : Calcul des features

## Fonctionnalités

1. **Sélection temporelle** : Date et heure de prédiction
2. **Conditions météo** : Impact de l'atm (atmosphère)
3. **Modes d'affichage** :
   - Top N points les plus risqués
   - Seuil de risque minimum
4. **Carte interactive** : Heatmap des zones à risque
5. **Top 10** : Liste des zones les plus dangereuses

## TODO Phase 3

- [ ] Intégrer chargement modèle depuis S3/DuckLake
- [ ] Connecter aux prédictions réelles (actuellement scores aléatoires)
- [ ] Ajouter cache pour les prédictions
- [ ] Améliorer la visualisation (clusters, markers)
- [ ] Ajouter statistiques détaillées
- [ ] Export des prédictions (CSV, GeoJSON)

## Dépendances

```bash
uv add streamlit streamlit-folium folium
```

Déjà installées via `pyproject.toml` optional-dependencies `[streamlit]`.
