import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import duckdb
    import polars as pl
    import pandas as pd
    import osmnx as ox
    import geopandas as gpd
    from shapely.ops import unary_union
    import os
    import numpy as np
    import glob
    import joblib
    import mlflow
    import mlflow.sklearn
    import mlflow.catboost
    import mlflow.lightgbm
    import mlflow.xgboost
    from dotenv import load_dotenv

    from shapely.geometry import Point

    # Charger les variables d'environnement depuis .env
    load_dotenv()
    print(f"🔑 Variables d'environnement chargées depuis .env")
    print(f"   AWS_ACCESS_KEY_ID: {'✅ défini' if os.getenv('AWS_ACCESS_KEY_ID') else '❌ manquant'}")
    print(f"   AWS_SECRET_ACCESS_KEY: {'✅ défini' if os.getenv('AWS_SECRET_ACCESS_KEY') else '❌ manquant'}")
    return Point, glob, gpd, joblib, mlflow, mo, np, os, ox, pd, pl


@app.cell
def _(mlflow):
    mlflow.set_tracking_uri("https://mlflow.tgu.ovh")
    mlflow.search_experiments()
    return


@app.cell
def _(mlflow):
    # Configuration MLflow
    mlflow.set_tracking_uri("https://mlflow.tgu.ovh")
    mlflow.set_experiment("accidents-nc")
    print(f"🔬 MLflow configuré : {mlflow.get_tracking_uri()}")
    print(f"📊 Expérience : {mlflow.get_experiment_by_name('accidents-nc').name}")
    return


@app.cell
def _():
    # Configuration optimisée pour meilleur recall
    CONFIG = {
        'n_negative_samples_ratio': 22000, 
        'buffer_meters': 200,
        'grid_step': 0.02,
        'test_size': 0.2,
        'random_state': 42,
        'cv_folds': 5,
        'search_iterations': 20,
        'accident_exclusion_buffer_km': 0.3,  # Réduit de 500m→300m
        'temporal_risk_ratio': 0.85  # 85% heures à risque (vs 70% avant)
    }
    return (CONFIG,)


@app.cell
def _(glob, mo):
    carac_files = glob.glob('data/caracteristiques-*.csv')

    if carac_files:
        print(f"📂 Chargement depuis les fichiers locaux (data/)")
        files_str = "[" + ", ".join([f"'{f}'" for f in sorted(carac_files)]) + "]"
        caracteristiques = mo.sql(
            f"""
            select * from read_csv(
                {files_str},
                union_by_name=true
            )
            """
        )
    else:
        print("⚠️  Fichiers locaux non trouvés, téléchargement depuis data.gouv.fr...")
        print("   Conseil: Exécutez './download_data.sh' pour accélérer le chargement")
        caracteristiques = mo.sql(
            f"""
            select * from read_csv([
                'https://www.data.gouv.fr/fr/datasets/r/e22ba475-45a3-46ac-a0f7-9ca9ed1e283a',
                'https://www.data.gouv.fr/fr/datasets/r/07a88205-83c1-4123-a993-cba5331e8ae0',
                'https://www.data.gouv.fr/fr/datasets/r/85cfdc0c-23e4-4674-9bcd-79a970d7269b',
                'https://www.data.gouv.fr/fr/datasets/r/5fc299c0-4598-4c29-b74c-6a67b0cc27e7',
                'https://www.data.gouv.fr/fr/datasets/r/104dbb32-704f-4e99-a71e-43563cb604f2',
                'https://www.data.gouv.fr/api/1/datasets/r/83f0fb0e-e0ef-47fe-93dd-9aaee851674a'
            ],union_by_name=true)
            """
        )
    return (caracteristiques,)


@app.cell
def _(glob, mo):

    # Charger depuis fichiers locaux si disponibles, sinon URLs distantes
    usagers_files = glob.glob('data/usagers-*.csv')

    if usagers_files:
        print(f"📂 Chargement depuis {len(usagers_files)} fichiers usagers locaux")
        files_str_usagers = "[" + ", ".join([f"'{f}'" for f in sorted(usagers_files)]) + "]"
        usagers = mo.sql(
            f"""
            select * from read_csv(
                {files_str_usagers},
                union_by_name=true
            )
            """
        )
    else:
        print("⚠️  Fichiers usagers non trouvés, téléchargement depuis data.gouv.fr...")
        usagers = mo.sql(
            f"""
            select * from read_csv([
                'https://www.data.gouv.fr/fr/datasets/r/36b1b7b3-84b4-4901-9163-59ae8a9e3028',
                'https://www.data.gouv.fr/fr/datasets/r/78c45763-d170-4d51-a881-e3147802d7ee',
                'https://www.data.gouv.fr/fr/datasets/r/ba5a1956-7e82-41b7-a602-89d7dd484d7a',
                'https://www.data.gouv.fr/fr/datasets/r/62c20524-d442-46f5-bfd8-982c59763ec8',
                'https://www.data.gouv.fr/fr/datasets/r/68848e2a-28dd-4efc-9d5f-d512f7dbe66f',
                'https://www.data.gouv.fr/api/1/datasets/r/f57b1f58-386d-4048-8f78-2ebe435df868'
            ],union_by_name=true)
            """
        )
    return (usagers,)


@app.cell
def _(caracteristiques, mo, usagers):
    accidents = mo.sql(
        f"""
        select -- grav,
            strptime(concat(jour, '/', mois, '/',  an, ' ', hrmn), '%d/%m/%Y %H:%M:%S') as datetime,
            cast(trim(lat).replace(',', '.') as float) as latitude, 
            cast(trim(long).replace(',', '.') as float) as longitude, 
            atm,
            hour(datetime) as hour,
            dayofweek(datetime) as dayofweek,
            month(datetime) as month,
            1 as target
            from caracteristiques c
        	left join usagers u on c.Num_Acc = u.Num_Acc
        where dep = '988'
        """
    )
    return (accidents,)


@app.cell
def _(accidents, pl):
    print(f"📊 Accidents en Nouvelle-Calédonie : {len(accidents)} lignes")
    print(f"🔍 Valeurs manquantes par colonne :")
    print(accidents.null_count())
    print(f"📅 Période : {accidents['datetime'].min()} → {accidents['datetime'].max()}")

    # Filtrer les lignes avec coordonnées invalides
    accidents_filtres = accidents.filter(
        (pl.col('latitude').is_not_null()) & 
        (pl.col('longitude').is_not_null()) &
        (pl.col('latitude').is_between(-23.0, -19.5)) &
        (pl.col('longitude').is_between(163.5, 168.0))
    )
    print(f"✅ Après filtrage : {len(accidents_filtres)} lignes valides")
    return (accidents_filtres,)


@app.cell
def _():
    areas = [
            "Boulouparis, New Caledonia",
            "Bourail, New Caledonia",
            "Canala, New Caledonia",
            "Dumbéa, New Caledonia",
            "Farino, New Caledonia",
            "Hienghène, New Caledonia",
            "Houaïlou, New Caledonia",
            "Île des Pins, New Caledonia",
            "Kaala-Gomen, New Caledonia",
            "Koné, New Caledonia",
            "Kouaoua, New Caledonia",
            "Koumac, New Caledonia",
            "La Foa, New Caledonia",
            "Lifou, New Caledonia",
            "Maré, New Caledonia",
            "Moindou, New Caledonia",
            "Mont-Dore, New Caledonia",
            "Nouméa, New Caledonia",
            "Ouégoa, New Caledonia",
            "Ouvéa, New Caledonia",
            "Païta, New Caledonia",
            "Poindimié, New Caledonia",
            "Ponérihouen, New Caledonia",
            "Poya, New Caledonia",
            "Sarraméa, New Caledonia",
            "Thio, New Caledonia",
            "Touho, New Caledonia",
            "Voh, New Caledonia",
            "Yaté, New Caledonia",
            "Belep, New Caledonia",
            "Îles Loyauté, New Caledonia",  # Généralisation
            "Province Nord, New Caledonia", # utile si communes échouent
            "Province Sud, New Caledonia"
        ]
    return (areas,)


@app.cell
def _(CONFIG, Point, areas, gpd, joblib, np, os, ox, pd):

    # PRIORITÉ 1: Charger les données pré-calculées si disponibles
    if os.path.exists('routes_with_features.pkl'):
        print("📦 Chargement de routes_with_features.pkl (pré-calculé)")
        routes_data = joblib.load('routes_with_features.pkl')

        # Convertir en DataFrame selon le type
        if isinstance(routes_data, pd.DataFrame):
            routes_df_temp = routes_data
        elif isinstance(routes_data, dict):
            # Si le dict contient la clé 'routes_grid', l'extraire directement
            if 'routes_grid' in routes_data:
                routes_df_temp = routes_data['routes_grid']
                print(f"   ✅ Extraction de routes_grid depuis le dict")
            # Sinon, vérifier le type des valeurs
            else:
                first_val = next(iter(routes_data.values()))
                if isinstance(first_val, np.ndarray):
                    if first_val.ndim == 2:
                        routes_df_temp = pd.DataFrame({
                            key: val[:, 0] if isinstance(val, np.ndarray) and val.ndim == 2 else val
                            for key, val in routes_data.items()
                        })
                    else:
                        routes_df_temp = pd.DataFrame(routes_data)
                elif isinstance(first_val, pd.Series):
                    routes_df_temp = pd.DataFrame({
                        key: val.values for key, val in routes_data.items()
                    })
                elif isinstance(first_val, pd.DataFrame):
                    # Prendre le premier DataFrame disponible
                    routes_df_temp = first_val
                else:
                    try:
                        routes_df_temp = pd.DataFrame(routes_data)
                    except:
                        routes_df_temp = pd.DataFrame({
                            key: np.asarray(val).flatten()
                            for key, val in routes_data.items()
                        })
        elif isinstance(routes_data, np.ndarray):
            if routes_data.ndim == 2:
                col_names = ['latitude', 'longitude', 'road_type', 'speed_limit', 'accident_density_5km', 'dist_to_noumea_km']
                routes_df_temp = pd.DataFrame(routes_data, columns=col_names[:routes_data.shape[1]])
            else:
                routes_df_temp = pd.DataFrame(routes_data)
        else:
            routes_df_temp = pd.DataFrame(routes_data)

        # Convertir en GeoDataFrame avec géométrie
        routes_grid_loaded = gpd.GeoDataFrame(
            routes_df_temp,
            geometry=gpd.points_from_xy(routes_df_temp['longitude'], routes_df_temp['latitude']),
            crs="EPSG:4326"
        )

        print(f"✅ {len(routes_grid_loaded)} points chargés avec features enrichies")
        print(f"   Features: {', '.join([c for c in routes_grid_loaded.columns if c != 'geometry'])}")

        # Charger aussi routes.nc pour compatibilité (utilisé plus tard)
        if os.path.exists('routes.nc'):
            routes = gpd.read_file('routes.nc')
        else:
            routes = None
    else:
        # FALLBACK: Régénérer la grille (ancien comportement)
        print("⚠️  routes_with_features.pkl non trouvé, génération de la grille...")
        print("   Conseil: Exécutez 'python precompute_density.py' pour accélérer l'entraînement")

        all_edges = []
        filepath="routes.nc"

        if os.path.exists(filepath):
            routes = gpd.read_file(filepath)
        else:
            for place in areas:
                try:
                    print(f"⬇️  {place}...")
                    G = ox.graph_from_place(place, network_type='drive')
                    edges = ox.graph_to_gdfs(G, nodes=False)
                    all_edges.append(edges)
                except Exception as e:
                    print(f"⚠️  Échec pour {place} : {e}")

            if not all_edges:
                raise RuntimeError("Aucune donnée n'a pu être téléchargée.")

            # Concaténer toutes les routes
            all_routes = gpd.GeoDataFrame(pd.concat(all_edges, ignore_index=True), crs=all_edges[0].crs)

            # Supprimer les doublons géométriques
            all_routes = all_routes.drop_duplicates(subset="geometry")

            # Enregistrer
            all_routes.to_file(filepath, driver="GeoJSON")
            print(f"✅ Routes enregistrées dans {filepath}")
            routes = all_routes

        # Créer la grille spatiale
        step = CONFIG['grid_step']
        buffer_meters = CONFIG['buffer_meters']
        lat_min, lat_max = -23.0, -19.5
        lon_min, lon_max = 163.5, 168.0
        lats = np.arange(lat_min, lat_max, step)
        lons = np.arange(lon_min, lon_max, step)
        grid = pd.DataFrame([(lat, lon) for lat in lats for lon in lons], columns=["latitude", "longitude"])
        grid["geometry"] = grid.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
        grid_gdf = gpd.GeoDataFrame(grid, geometry="geometry", crs="EPSG:4326").to_crs(epsg=3857)

        # Buffer autour des routes
        routes_buffer = routes.to_crs(epsg=3857).buffer(buffer_meters)
        routes_buffer_gdf = gpd.GeoDataFrame(geometry=routes_buffer, crs="EPSG:3857")

        # Spatial join
        grid_on_roads = gpd.sjoin(grid_gdf, routes_buffer_gdf, how="inner", predicate="intersects").drop(columns="index_right")
        routes_grid_loaded = grid_on_roads.to_crs(epsg=4326)
        print(f"🗺️  Grille spatiale : {len(routes_grid_loaded)} points sur routes (sans features enrichies)")

    # Assigner la variable finale (garantit qu'elle existe toujours en tant que GeoDataFrame)
    routes_grid = routes_grid_loaded
    return (routes_grid,)


@app.cell
def _(CONFIG, accidents, accidents_filtres, gpd, np, pd, pl, routes_grid):
    # ÉTAPE 1 : Calculer le nombre d'échantillons négatifs (ratio 3:1)
    n_samples = len(accidents_filtres) * CONFIG['n_negative_samples_ratio']
    # Ex: 1000 accidents × 3 = 3000 échantillons négatifs

    # ÉTAPE 2 : Vérifier qu'on a des données valides
    if len(routes_grid) > 0 and len(accidents_filtres) > 0:
        print("🎯 Stratégie : exclusion spatiale des zones d'accidents")

        # ─────────────────────────────────────────────────────────
        # PARTIE A : PRÉPARATION DES DONNÉES GÉOSPATIALES
        # ─────────────────────────────────────────────────────────

        # Convertir les accidents en GeoDataFrame (objets géospatiaux)
        accidents_pd = accidents_filtres.to_pandas()
        # Fix pandas 2.x/pyarrow: convertir toutes les colonnes en numpy avant GeoDataFrame
        accidents_data = {}
        for col in accidents_pd.columns:
            if hasattr(accidents_pd[col], 'to_numpy'):
                accidents_data[col] = accidents_pd[col].to_numpy()
            else:
                accidents_data[col] = accidents_pd[col].values

        accidents_gdf = gpd.GeoDataFrame(
            accidents_data,
            geometry=gpd.points_from_xy(
                accidents_data['longitude'],
                accidents_data['latitude']
            ),
            crs="EPSG:4326"  # Coordonnées géographiques standard
        )
        # Résultat : Points géolocalisés des accidents

        # ─────────────────────────────────────────────────────────
        # PARTIE B : CRÉER DES ZONES D'EXCLUSION AUTOUR DES ACCIDENTS
        # ─────────────────────────────────────────────────────────

        buffer_km = CONFIG['accident_exclusion_buffer_km']  # 0.5 km = 500m

        # Reprojection en EPSG:3857 (Web Mercator) pour travailler en mètres
        accidents_buffer = accidents_gdf.to_crs(epsg=3857).buffer(buffer_km * 1000)
        # .buffer(500) crée un cercle de 500m autour de chaque accident

        accidents_buffer_gdf = gpd.GeoDataFrame(geometry=accidents_buffer, crs="EPSG:3857")

        # Visualisation conceptuelle :
        #   🔴 = accident
        #   ⭕ = zone d'exclusion de 500m de rayon
        #
        #        ⭕⭕⭕
        #      ⭕⭕🔴⭕⭕
        #        ⭕⭕⭕
        #
        # On NE VEUT PAS générer de négatifs dans ces cercles !

        # ─────────────────────────────────────────────────────────
        # PARTIE C : PRÉPARER LA GRILLE DES ROUTES
        # ─────────────────────────────────────────────────────────

        # routes_grid est déjà un GeoDataFrame, juste le reprojeter
        # Fix pandas 2.x/pyarrow: reconstruire avec colonnes numpy
        routes_grid_data = {}
        for col in routes_grid.columns:
            if col != 'geometry':
                if hasattr(routes_grid[col], 'to_numpy'):
                    routes_grid_data[col] = routes_grid[col].to_numpy()
                else:
                    routes_grid_data[col] = routes_grid[col].values

        routes_grid_fixed = gpd.GeoDataFrame(
            routes_grid_data,
            geometry=routes_grid.geometry.values,
            crs=routes_grid.crs
        )
        routes_grid_gdf = routes_grid_fixed.to_crs(epsg=3857)

        # ─────────────────────────────────────────────────────────
        # PARTIE D : FILTRAGE SPATIAL - TROUVER LES ROUTES SÛRES
        # ─────────────────────────────────────────────────────────

        # Spatial join : associer chaque point de route avec les zones d'accidents
        safe_routes = routes_grid_gdf.sjoin(
            accidents_buffer_gdf,
            how="left",           # Garder tous les points de route
            predicate="intersects" # Marquer ceux qui intersectent une zone d'accident
        )
        # 'index_right' sera NaN pour les points qui n'intersectent AUCUNE zone d'accident

        # Filtrer : garder SEULEMENT les points avec index_right = NaN
        safe_routes = safe_routes[safe_routes['index_right'].isna()].drop(columns=['index_right'])

        # Reconvertir en coordonnées géographiques standard
        safe_routes = safe_routes.to_crs(epsg=4326)

        print(f"   📍 Points disponibles : {len(routes_grid)} total, {len(safe_routes)} hors zones d'accidents")
        # Ex: 50 000 points sur routes → 35 000 points "sûrs" (hors zones d'accidents)

        # ─────────────────────────────────────────────────────────
        # PARTIE E : ÉCHANTILLONNAGE SPATIAL
        # ─────────────────────────────────────────────────────────

        if len(safe_routes) > 0:
            # Échantillonner aléatoirement depuis les zones sûres
            route_sample_indices = np.random.choice(
                len(safe_routes),
                size=min(n_samples, len(safe_routes) * 10),  # Limiter si pas assez de points
                replace=True  # Permettre de réutiliser les mêmes points
            )
            route_sample = safe_routes.iloc[route_sample_indices]
            # Résultat : 3000 points de coordonnées (lat, lon) sur routes SÛRES

            # ─────────────────────────────────────────────────────────
            # PARTIE F : STRATÉGIE TEMPORELLE HYBRIDE OPTIMISÉE
            # ─────────────────────────────────────────────────────────

            # Répartition : 85% heures à risque + 15% heures aléatoires (optimisé)
            n_from_accidents = int(len(route_sample) * CONFIG['temporal_risk_ratio'])
            n_random = len(route_sample) - n_from_accidents

            # Générer 30% de timestamps uniformément distribués sur toute la période
            date_range = accidents['datetime'].max() - accidents['datetime'].min()
            # Ex: 5 ans = 157 680 000 secondes

            random_timestamps = [
                accidents['datetime'].min() + np.timedelta64(
                    int(np.random.uniform(0, date_range.total_seconds())), 
                    's'
                )
                for _ in range(n_random)
            ]
            # Génère des timestamps aléatoires uniformément distribués

            # Combiner : 85% depuis accidents + 15% aléatoires (optimisé pour recall)
            combined_timestamps = np.concatenate([
                np.random.choice(accidents['datetime'], size=n_from_accidents),
                random_timestamps
            ])
            np.random.shuffle(combined_timestamps)  # Mélanger pour éviter les patterns

            # ─────────────────────────────────────────────────────────
            # PARTIE G : CONSTRUIRE LE DATAFRAME FINAL
            # ─────────────────────────────────────────────────────────

            sampled = pl.DataFrame({
                'datetime': pd.to_datetime(combined_timestamps),  # Normaliser les timestamps
                'latitude': route_sample['latitude'].values,    # Coordonnées SÛRES
                'longitude': route_sample['longitude'].values,  # (hors zones d'accidents)
                'atm': np.random.choice(accidents['atm'], size=len(route_sample)),  # Conditions météo
            })

        else:
            # FALLBACK 1 : Si pas assez de zones sûres, utiliser toute la grille
            print("⚠️  Pas assez de zones sûres, utilisation de la grille complète")
            route_sample_indices = np.random.choice(
                len(routes_grid),
                size=min(n_samples, len(routes_grid) * 10),
                replace=True
            )
            route_sample = routes_grid.iloc[route_sample_indices]
            sampled = pl.DataFrame({
                'datetime': pd.to_datetime(np.random.choice(accidents['datetime'], size=len(route_sample))),
                'latitude': route_sample['latitude'].values,
                'longitude': route_sample['longitude'].values,
                'atm': np.random.choice(accidents['atm'], size=len(route_sample)),
            })

    else:
        # FALLBACK 2 : Si pas de grille de routes du tout
        print("⚠️  Grille vide, utilisation des coordonnées des accidents")
        sampled = pl.DataFrame({
            'datetime': pd.to_datetime(np.random.choice(accidents['datetime'], size=n_samples)),
            'latitude': np.random.choice(accidents['latitude'], size=n_samples),
            'longitude': np.random.choice(accidents['longitude'], size=n_samples),
            'atm': np.random.choice(accidents['atm'], size=n_samples),
        })

    # ─────────────────────────────────────────────────────────
    # PARTIE H : EXTRACTION DES FEATURES TEMPORELLES
    # ─────────────────────────────────────────────────────────

    sampled = sampled.with_columns([
        pl.col('datetime').dt.hour().alias('hour'),           # Heure (0-23)
        pl.col('datetime').dt.weekday().alias('dayofweek'),   # Jour semaine (0-6)
        pl.col('datetime').dt.month().alias('month'),         # Mois (1-12)
        pl.lit(0).alias('target')  # LABEL = 0 (pas d'accident)
    ])

    negative_samples = sampled
    print(f"🎲 Échantillons négatifs générés : {len(negative_samples)} (ratio {CONFIG['n_negative_samples_ratio']}:1)")
    negative_samples
    return (negative_samples,)


@app.cell
def _(accidents_filtres, mo, negative_samples, np, pl, routes_grid):
    import warnings
    from scipy.spatial.distance import cdist
    warnings.filterwarnings('ignore')

    print("\n🔧 ENRICHISSEMENT DES FEATURES")

    # Combiner accidents et négatifs temporairement
    full_dataset_temp = mo.sql(
        f"""
        select * from accidents_filtres
        union
        select * from negative_samples
        """
    ).to_pandas()

    # ==========================================
    # 1. FEATURES D INTERACTION SPATIO-TEMPORELLES
    # ==========================================
    print("1️⃣ Features d'interaction spatio-temporelles...")

    # Normaliser les interactions
    full_dataset_temp['lat_hour'] = full_dataset_temp['latitude'] * full_dataset_temp['hour'] / 24
    full_dataset_temp['lon_hour'] = full_dataset_temp['longitude'] * full_dataset_temp['hour'] / 24
    full_dataset_temp['lat_dayofweek'] = full_dataset_temp['latitude'] * full_dataset_temp['dayofweek'] / 7
    full_dataset_temp['lon_dayofweek'] = full_dataset_temp['longitude'] * full_dataset_temp['dayofweek'] / 7

    # Features temporelles binaires
    full_dataset_temp['is_weekend'] = (full_dataset_temp['dayofweek'] >= 5).astype(int)
    full_dataset_temp['is_rush_morning'] = ((full_dataset_temp['hour'] >= 6) & (full_dataset_temp['hour'] <= 8)).astype(int)
    full_dataset_temp['is_rush_evening'] = ((full_dataset_temp['hour'] >= 16) & (full_dataset_temp['hour'] <= 18)).astype(int)
    full_dataset_temp['is_night'] = ((full_dataset_temp['hour'] >= 19) | (full_dataset_temp['hour'] <= 5)).astype(int)

    # Encodage cyclique (éviter discontinuité 23h→0h)
    full_dataset_temp['hour_sin'] = np.sin(2 * np.pi * full_dataset_temp['hour'] / 24)
    full_dataset_temp['hour_cos'] = np.cos(2 * np.pi * full_dataset_temp['hour'] / 24)
    full_dataset_temp['dayofweek_sin'] = np.sin(2 * np.pi * full_dataset_temp['dayofweek'] / 7)
    full_dataset_temp['dayofweek_cos'] = np.cos(2 * np.pi * full_dataset_temp['dayofweek'] / 7)

    # ==========================================
    # 2. ENRICHISSEMENT OSM + DENSITÉ (PRÉ-CALCULÉS)
    # ==========================================
    print("2️⃣ Enrichissement OSM + densité (depuis routes_with_features.pkl)...")

    # Vérifier si routes_grid a les features pré-calculées
    has_precomputed = all(col in routes_grid.columns for col in ['road_type', 'speed_limit', 'accident_density_5km', 'dist_to_noumea_km'])

    if has_precomputed:
        print("   ✅ Utilisation des features pré-calculées")
        # Trouver la grille la plus proche pour chaque observation
        full_coords = full_dataset_temp[['latitude', 'longitude']].values
        grid_coords = routes_grid[['latitude', 'longitude']].values

        # Fix pandas 2.x/pyarrow: convertir les colonnes en numpy arrays une seule fois
        road_type_array = routes_grid['road_type'].to_numpy() if hasattr(routes_grid['road_type'], 'to_numpy') else routes_grid['road_type'].values
        speed_limit_array = routes_grid['speed_limit'].to_numpy() if hasattr(routes_grid['speed_limit'], 'to_numpy') else routes_grid['speed_limit'].values
        density_array = routes_grid['accident_density_5km'].to_numpy() if hasattr(routes_grid['accident_density_5km'], 'to_numpy') else routes_grid['accident_density_5km'].values
        dist_noumea_array = routes_grid['dist_to_noumea_km'].to_numpy() if hasattr(routes_grid['dist_to_noumea_km'], 'to_numpy') else routes_grid['dist_to_noumea_km'].values

        # Calculer distances (par batch pour économiser la mémoire)
        batch_size = 1000
        road_types = []
        speed_limits = []
        densities = []
        distances_noumea = []

        for i in range(0, len(full_coords), batch_size):
            batch = full_coords[i:i+batch_size]
            distances = cdist(batch, grid_coords)
            nearest_indices = distances.argmin(axis=1)

            road_types.extend(road_type_array[nearest_indices])
            speed_limits.extend(speed_limit_array[nearest_indices])
            densities.extend(density_array[nearest_indices])
            distances_noumea.extend(dist_noumea_array[nearest_indices])

        full_dataset_temp['road_type'] = road_types
        full_dataset_temp['speed_limit'] = speed_limits
        full_dataset_temp['accident_density_5km'] = densities
        full_dataset_temp['dist_to_noumea_km'] = distances_noumea
    else:
        print("   ⚠️  Features pré-calculées non disponibles, valeurs par défaut")
        full_dataset_temp['road_type'] = 2
        full_dataset_temp['speed_limit'] = 50
        full_dataset_temp['accident_density_5km'] = 0
        full_dataset_temp['dist_to_noumea_km'] = 100

    # ==========================================
    # 3. FEATURES TEMPORELLES AVANCÉES
    # ==========================================
    print("3️⃣ Features temporelles avancées...")

    # Jours fériés Nouvelle-Calédonie (approximation simplifiée)
    nc_holidays_days = [
        (1, 1), (5, 1), (5, 8), (7, 14),
        (9, 24), (11, 1), (11, 11), (12, 25)
    ]

    def is_holiday(row):
        month = row['month']
        day = row['datetime'].day
        return 1 if (month, day) in nc_holidays_days else 0

    full_dataset_temp['is_holiday'] = full_dataset_temp.apply(is_holiday, axis=1)

    # Vacances scolaires (approximation: janvier, juillet-août, décembre)
    full_dataset_temp['school_holidays'] = full_dataset_temp['month'].isin([1, 7, 8, 12]).astype(int)

    print(f"\n✅ Features enrichies : {len(full_dataset_temp.columns)} colonnes")
    print(f"   Dont {4} features pré-calculées (OSM + densité)")

    full_dataset = pl.from_pandas(full_dataset_temp)
    return (full_dataset,)


@app.cell
def _():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.base import ClassifierMixin
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from imblearn.ensemble import BalancedRandomForestClassifier
    from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score, roc_auc_score
    import time
    import optuna
    from optuna.integration import CatBoostPruningCallback, LightGBMPruningCallback
    return (
        CatBoostClassifier,
        CatBoostPruningCallback,
        LGBMClassifier,
        LabelEncoder,
        LightGBMPruningCallback,
        XGBClassifier,
        f1_score,
        optuna,
        precision_score,
        recall_score,
        roc_auc_score,
        time,
        train_test_split,
    )


@app.cell
def _(CONFIG):
    test_size = CONFIG['test_size']
    random_state = CONFIG['random_state']
    return random_state, test_size


@app.cell
def _(LabelEncoder, full_dataset, random_state, test_size, train_test_split):
    le = LabelEncoder()
    tmp_dataset = full_dataset.to_pandas()
    tmp_dataset['atm'] = le.fit_transform(tmp_dataset['atm'])

    # Features enrichies avec les nouvelles colonnes
    features = [
        # Géographiques originales
        'latitude', 'longitude',
        # Temporelles originales
        'hour', 'dayofweek', 'month',
        # Météo originale
        'atm',
        # 1. Interactions spatio-temporelles
        'lat_hour', 'lon_hour', 'lat_dayofweek', 'lon_dayofweek',
        'is_weekend', 'is_rush_morning', 'is_rush_evening', 'is_night',
        'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
        # 2. Caractéristiques routes OSM
        'road_type', 'speed_limit',
        # 3. Densité et proximité
        'accident_density_5km', 'dist_to_noumea_km',
        # 4. Temporelles avancées
        'is_holiday', 'school_holidays'
    ]

    print(f"\n📊 FEATURES UTILISÉES : {len(features)} features")
    print(f"   - Géographiques: 2 (lat, lon)")
    print(f"   - Temporelles originales: 3 (hour, dayofweek, month)")  
    print(f"   - Météo: 1 (atm)")
    print(f"   - Interactions spatio-temporelles: 12")
    print(f"   - Routes OSM: 2")
    print(f"   - Densité/proximité: 2")
    print(f"   - Temporelles avancées: 2")

    dataset = tmp_dataset.dropna(subset=features)
    X = dataset[features] 
    y = dataset['target']
    # stratify=y, 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print(f"\n📈 DATASET FINAL")
    print(f"   - Total: {len(dataset)} observations")
    print(f"   - Train: {len(X_train)} ({len(X_train)/len(dataset):.1%})")
    print(f"   - Test: {len(X_test)} ({len(X_test)/len(dataset):.1%})")
    print(f"   - Ratio accidents: {y.sum()}/{len(y)} ({y.sum()/len(y):.2%})")
    return X_test, X_train, features, le, y_test, y_train


@app.cell
def _(
    CONFIG,
    CatBoostClassifier,
    CatBoostPruningCallback,
    LGBMClassifier,
    LightGBMPruningCallback,
    XGBClassifier,
    X_test,
    X_train,
    f1_score,
    mlflow,
    optuna,
    pd,
    precision_score,
    recall_score,
    roc_auc_score,
    time,
    y_test,
    y_train,
):


    print("🔧 TUNING HYPERPARAMÈTRES (Optuna)\n")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Calculer scale_weight pour classes déséquilibrées
    tuning_n_negative = (y_train == 0).sum()
    tuning_n_positive = (y_train == 1).sum()
    tuning_scale_weight = tuning_n_negative / tuning_n_positive
    print(f"⚖️  Ratio classes: {tuning_scale_weight:.1f}:1 (négatif:positif)\n")

    # Créer un run parent MLflow pour toute la session de tuning
    mlflow.start_run(run_name="hyperparameter-tuning")
    mlflow.log_params({
        'n_negative_samples_ratio': CONFIG['n_negative_samples_ratio'],
        'buffer_meters': CONFIG['buffer_meters'],
        'grid_step': CONFIG['grid_step'],
        'test_size': CONFIG['test_size'],
        'scale_weight': tuning_scale_weight,
        'n_train_samples': len(y_train),
        'n_test_samples': len(y_test)
    })

    # ========================================
    # 1. CATBOOST TUNING
    # ========================================
    def objective_catboost(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 150, 500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'auto_class_weights': 'Balanced',
            'eval_metric': 'AUC',  # Nécessaire pour le callback
            'random_state': CONFIG['random_state'],
            'verbose': False
        }

        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            callbacks=[CatBoostPruningCallback(trial, 'AUC')],
            early_stopping_rounds=50,
            verbose=False
        )

        y_pred = model.predict(X_test)
        recall = recall_score(y_test, y_pred, pos_label=1)
        return recall

    study_catboost = optuna.create_study(
        direction='maximize',
        study_name='CatBoost',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    print("🐱 CatBoost tuning (50 trials)...")
    start = time.time()
    study_catboost.optimize(objective_catboost, n_trials=50, show_progress_bar=False)
    catboost_time = time.time() - start
    print(f"   ✅ Meilleur Recall: {study_catboost.best_value:.3f} ({catboost_time:.0f}s)\n")

    # ========================================
    # 2. LIGHTGBM TUNING
    # ========================================
    def objective_lightgbm(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 150, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'is_unbalance': True,
            'metric': 'auc',  # Nécessaire pour le callback
            'random_state': CONFIG['random_state'],
            'verbose': -1
        }

        model = LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[LightGBMPruningCallback(trial, 'auc')]
        )

        y_pred = model.predict(X_test)
        recall = recall_score(y_test, y_pred, pos_label=1)
        return recall

    study_lgbm = optuna.create_study(
        direction='maximize',
        study_name='LightGBM',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    print("💡 LightGBM tuning (50 trials)...")
    start = time.time()
    study_lgbm.optimize(objective_lightgbm, n_trials=50, show_progress_bar=False)
    lgbm_time = time.time() - start
    print(f"   ✅ Meilleur Recall: {study_lgbm.best_value:.3f} ({lgbm_time:.0f}s)\n")

    # ========================================
    # 3. XGBOOST TUNING
    # ========================================
    def objective_xgboost(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 150, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': tuning_scale_weight,
            'random_state': CONFIG['random_state'],
            'verbosity': 0
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        recall = recall_score(y_test, y_pred, pos_label=1)
        return recall

    study_xgb = optuna.create_study(
        direction='maximize',
        study_name='XGBoost',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    print("⚡ XGBoost tuning (50 trials)...")
    start = time.time()
    study_xgb.optimize(objective_xgboost, n_trials=50, show_progress_bar=False)
    xgb_time = time.time() - start
    print(f"   ✅ Meilleur Recall: {study_xgb.best_value:.3f} ({xgb_time:.0f}s)\n")

    # ========================================
    # 4. ENTRAÎNER LES MODÈLES OPTIMAUX
    # ========================================
    print("\n🏆 ENTRAÎNEMENT DES MODÈLES OPTIMISÉS\n")

    tuned_models = {}
    tuning_results = []
    model_run_ids = {}  # Stocker les run_id de chaque modèle

    # CatBoost
    print("🐱 CatBoost final...")
    with mlflow.start_run(run_name="CatBoost", nested=True) as run:
        model_run_ids['CatBoost'] = run.info.run_id

        best_catboost = CatBoostClassifier(
            **study_catboost.best_params,
            auto_class_weights='Balanced',
            eval_metric='AUC',
            random_state=CONFIG['random_state'],
            verbose=False
        )
        best_catboost.fit(X_train, y_train)
        y_pred_cat = best_catboost.predict(X_test)
        y_proba_cat = best_catboost.predict_proba(X_test)[:, 1]

        recall_cat = recall_score(y_test, y_pred_cat, pos_label=1)
        precision_cat = precision_score(y_test, y_pred_cat, pos_label=1)
        f1_cat = f1_score(y_test, y_pred_cat, pos_label=1)
        auc_cat = roc_auc_score(y_test, y_proba_cat)

        # Logger dans MLflow
        mlflow.log_params(study_catboost.best_params)
        mlflow.log_metrics({
            'recall': recall_cat,
            'precision': precision_cat,
            'f1_score': f1_cat,
            'auc_roc': auc_cat,
            'training_time_seconds': catboost_time
        })
        mlflow.catboost.log_model(cb_model=best_catboost, name="model")

        tuned_models['CatBoost'] = best_catboost
        tuning_results.append({
            'Modèle': 'CatBoost',
            'Recall': f"{recall_cat:.3f}",
            'Precision': f"{precision_cat:.3f}",
            'F1-Score': f"{f1_cat:.3f}",
            'AUC-ROC': f"{auc_cat:.3f}",
            'Temps (s)': f"{catboost_time:.0f}",
            '_recall_raw': recall_cat
        })
        print(f"   Recall: {recall_cat:.3f} | F1: {f1_cat:.3f} | AUC: {auc_cat:.3f}")

    # LightGBM
    print("💡 LightGBM final...")
    with mlflow.start_run(run_name="LightGBM", nested=True) as run:
        model_run_ids['LightGBM'] = run.info.run_id

        best_lgbm = LGBMClassifier(
            **study_lgbm.best_params,
            is_unbalance=True,
            metric='auc',
            random_state=CONFIG['random_state'],
            verbose=-1
        )
        best_lgbm.fit(X_train, y_train)
        y_pred_lgbm = best_lgbm.predict(X_test)
        y_proba_lgbm = best_lgbm.predict_proba(X_test)[:, 1]

        recall_lgbm = recall_score(y_test, y_pred_lgbm, pos_label=1)
        precision_lgbm = precision_score(y_test, y_pred_lgbm, pos_label=1)
        f1_lgbm = f1_score(y_test, y_pred_lgbm, pos_label=1)
        auc_lgbm = roc_auc_score(y_test, y_proba_lgbm)

        # Logger dans MLflow
        mlflow.log_params(study_lgbm.best_params)
        mlflow.log_metrics({
            'recall': recall_lgbm,
            'precision': precision_lgbm,
            'f1_score': f1_lgbm,
            'auc_roc': auc_lgbm,
            'training_time_seconds': lgbm_time
        })
        mlflow.lightgbm.log_model(lgb_model=best_lgbm, name="model")

        tuned_models['LightGBM'] = best_lgbm
        tuning_results.append({
            'Modèle': 'LightGBM',
            'Recall': f"{recall_lgbm:.3f}",
            'Precision': f"{precision_lgbm:.3f}",
            'F1-Score': f"{f1_lgbm:.3f}",
            'AUC-ROC': f"{auc_lgbm:.3f}",
            'Temps (s)': f"{lgbm_time:.0f}",
            '_recall_raw': recall_lgbm
        })
        print(f"   Recall: {recall_lgbm:.3f} | F1: {f1_lgbm:.3f} | AUC: {auc_lgbm:.3f}")

    # XGBoost
    print("⚡ XGBoost final...")
    with mlflow.start_run(run_name="XGBoost", nested=True) as run:
        model_run_ids['XGBoost'] = run.info.run_id

        best_xgb = XGBClassifier(
            **study_xgb.best_params,
            scale_pos_weight=tuning_scale_weight,
            random_state=CONFIG['random_state'],
            verbosity=0
        )
        best_xgb.fit(X_train, y_train)
        y_pred_xgb = best_xgb.predict(X_test)
        y_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]

        recall_xgb = recall_score(y_test, y_pred_xgb, pos_label=1)
        precision_xgb = precision_score(y_test, y_pred_xgb, pos_label=1)
        f1_xgb = f1_score(y_test, y_pred_xgb, pos_label=1)
        auc_xgb = roc_auc_score(y_test, y_proba_xgb)

        # Logger dans MLflow
        mlflow.log_params(study_xgb.best_params)
        mlflow.log_metrics({
            'recall': recall_xgb,
            'precision': precision_xgb,
            'f1_score': f1_xgb,
            'auc_roc': auc_xgb,
            'training_time_seconds': xgb_time
        })
        mlflow.xgboost.log_model(xgb_model=best_xgb, name="model")

        tuned_models['XGBoost'] = best_xgb
        tuning_results.append({
            'Modèle': 'XGBoost',
            'Recall': f"{recall_xgb:.3f}",
            'Precision': f"{precision_xgb:.3f}",
            'F1-Score': f"{f1_xgb:.3f}",
            'AUC-ROC': f"{auc_xgb:.3f}",
            'Temps (s)': f"{xgb_time:.0f}",
            '_recall_raw': recall_xgb
        })
        print(f"   Recall: {recall_xgb:.3f} | F1: {f1_xgb:.3f} | AUC: {auc_xgb:.3f}")

    # Tableau récapitulatif
    tuning_results_df_full = pd.DataFrame(tuning_results).sort_values('_recall_raw', ascending=False)
    best_recall_value = tuning_results_df_full.iloc[0]['_recall_raw']
    tuning_results_df = tuning_results_df_full.drop(columns='_recall_raw')
    print("\n📊 RÉSULTATS FINAUX (trié par Recall)\n")
    print(tuning_results_df.to_string(index=False))

    # Sélectionner le meilleur
    tuned_best_model_name = tuning_results_df.iloc[0]['Modèle']
    tuned_best_model = tuned_models[tuned_best_model_name]

    print(f"\n🥇 GAGNANT : {tuned_best_model_name}")
    print(f"   → Meilleur Recall pour détecter les accidents\n")

    # Logger le meilleur modèle dans le run parent
    mlflow.log_metric("best_recall", best_recall_value)
    mlflow.log_param("best_model", tuned_best_model_name)
    mlflow.log_table(tuning_results_df, "comparison_table.json")

    # Enregistrer le meilleur modèle dans le Model Registry
    best_model_run_id = model_run_ids[tuned_best_model_name]
    model_uri = f"runs:/{best_model_run_id}/model"
    model_details = mlflow.register_model(
        model_uri=model_uri,
        name="accidents-nc-best-model"
    )
    print(f"\n📦 Modèle enregistré dans MLflow Registry: {model_details.name} (version {model_details.version})")

    # Fermer le run parent
    mlflow.end_run()

    # Stocker les études pour visualisation
    optuna_studies = {
        'CatBoost': study_catboost,
        'LightGBM': study_lgbm,
        'XGBoost': study_xgb
    }
    return (
        optuna_studies,
        tuned_best_model,
        tuned_best_model_name,
        tuned_models,
    )


@app.cell
def _(mo, optuna_studies):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    print("📊 VISUALISATION OPTUNA\n")

    # Sélectionner le modèle à visualiser
    model_selector = mo.ui.dropdown(
        options=list(optuna_studies.keys()),
        value='CatBoost',
        label="Modèle à visualiser"
    )

    model_selector
    return (model_selector,)


@app.cell
def _(model_selector, optuna_studies):
    import optuna.visualization as vis

    selected_study = optuna_studies[model_selector.value]

    # 1. Historique d'optimisation
    fig_history = vis.plot_optimization_history(selected_study)
    fig_history.update_layout(
        title=f'Historique Optimisation - {model_selector.value}',
        height=400
    )

    # 2. Importance des hyperparamètres
    fig_importance = vis.plot_param_importances(selected_study)
    fig_importance.update_layout(
        title=f'Importance Hyperparamètres - {model_selector.value}',
        height=400
    )

    # 3. Coordonnées parallèles
    fig_parallel = vis.plot_parallel_coordinate(selected_study)
    fig_parallel.update_layout(
        title=f'Coordonnées Parallèles - {model_selector.value}',
        height=500
    )

    print(f"\n📈 Meilleurs paramètres pour {model_selector.value}:")
    for param, value in selected_study.best_params.items():
        print(f"   {param}: {value}")

    [fig_history, fig_importance, fig_parallel]
    return


@app.cell
def _(X_test, features, pd, tuned_best_model, tuned_best_model_name, y_test):
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    import matplotlib.pyplot as plt

    # Prédictions
    tuned_y_pred = tuned_best_model.predict(X_test)
    tuned_y_proba = tuned_best_model.predict_proba(X_test)[:, 1]

    print(f"📊 RAPPORT DÉTAILLÉ - {tuned_best_model_name}\n")

    # Rapport de classification
    print("Classification :")
    print(classification_report(y_test, tuned_y_pred, target_names=['Pas accident', 'Accident']))

    # Importance des features (si disponible)
    if hasattr(tuned_best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': tuned_best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n🎯 Importance des features :")
        print(feature_importance.to_string(index=False))
    else:
        print("\n⚠️ Importance des features non disponible pour ce modèle")

    # Matrice de confusion
    cm = confusion_matrix(y_test, tuned_y_pred)
    print(f"\n🔢 Matrice de confusion :")
    print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

    # Calcul des taux
    best_recall_accident = cm[1,1] / (cm[1,0] + cm[1,1])
    print(f"\n✅ Accidents détectés : {cm[1,1]}/{cm[1,0] + cm[1,1]} ({best_recall_accident:.1%})")
    print(f"❌ Accidents ratés : {cm[1,0]} ({(1-best_recall_accident):.1%})")

    # Courbe ROC
    fpr, tpr, thresholds = roc_curve(y_test, tuned_y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'{tuned_best_model_name} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Courbe ROC - {tuned_best_model_name}')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(
    features,
    joblib,
    le,
    tuned_best_model,
    tuned_best_model_name,
    tuned_models,
):

    # Sauvegarder le meilleur modèle
    joblib.dump(tuned_best_model, 'accident_model.pkl')
    joblib.dump(le, 'atm_encoder.pkl')
    joblib.dump(features, 'features.pkl')
    joblib.dump({'name': tuned_best_model_name, 'model': tuned_best_model}, 'best_model_info.pkl')

    # Sauvegarder tous les modèles pour comparaison future
    joblib.dump(tuned_models, 'all_models.pkl')

    print(f"💾 Meilleur modèle ({tuned_best_model_name}) sauvegardé dans accident_model.pkl")
    print("💾 Encodeur sauvegardé dans atm_encoder.pkl")
    print("💾 Liste des features sauvegardée dans features.pkl")
    print("💾 Tous les modèles sauvegardés dans all_models.pkl")

    # Info pour la production
    print("\n📝 Pour charger le modèle en production :")
    print("model = joblib.load('accident_model.pkl')")
    print("encoder = joblib.load('atm_encoder.pkl')")
    print("features = joblib.load('features.pkl')")
    return


if __name__ == "__main__":
    app.run()
