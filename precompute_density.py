#!/usr/bin/env python3
"""
Pré-calcule la densité d'accidents historiques pour chaque point de la grille routière.
Évite le calcul coûteux à chaque prédiction.
"""

import os
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors
from geopy.distance import geodesic
import duckdb


def load_accidents_data():
    """Charge les accidents historiques depuis fichiers CSV locaux"""
    print("📥 Chargement des accidents historiques...")
    
    # Vérifier si le dossier data existe
    if not os.path.exists('data'):
        print("❌ Dossier 'data/' introuvable.")
        print("💡 Exécutez d'abord: ./download_data.sh")
        raise FileNotFoundError("Dossier 'data/' manquant. Lancez ./download_data.sh")
    
    # Lister les fichiers caractéristiques disponibles
    import glob
    csv_files = glob.glob('data/caracteristiques-*.csv')
    
    if len(csv_files) == 0:
        print("❌ Aucun fichier CSV trouvé dans data/")
        print("💡 Exécutez: ./download_data.sh")
        raise FileNotFoundError("Fichiers CSV manquants")
    
    print(f"   Trouvé {len(csv_files)} fichiers CSV")
    for f in sorted(csv_files):
        print(f"   - {os.path.basename(f)}")
    
    conn = duckdb.connect()
    
    # Construire la liste des chemins pour read_csv
    csv_paths_str = "[" + ", ".join([f"'{path}'" for path in csv_files]) + "]"
    
    query = f"""
    WITH caracteristiques AS (
        SELECT 
            "Num_Acc" AS num_acc,
            dep,
            CAST(TRIM(REPLACE(lat, ',', '.')) AS DOUBLE) AS latitude,
            CAST(TRIM(REPLACE(long, ',', '.')) AS DOUBLE) AS longitude,
            TRY_STRPTIME(
                concat(
                    LPAD(CAST(jour AS VARCHAR), 2, '0'), '/',
                    LPAD(CAST(mois AS VARCHAR), 2, '0'), '/',
                    CAST(an AS VARCHAR), ' ',
                    hrmn
                ),
                '%d/%m/%Y %H:%M:%S'
            ) AS datetime_accident,
            COALESCE(atm, 1) AS atm
        FROM read_csv(
            {csv_paths_str},
            union_by_name=true,
            ignore_errors=true,
            all_varchar=false,
            filename=true
        )
        WHERE dep = '988'
          AND lat IS NOT NULL
          AND long IS NOT NULL
          AND hrmn IS NOT NULL
          AND CAST(TRIM(REPLACE(lat, ',', '.')) AS DOUBLE) BETWEEN -23 AND -19
          AND CAST(TRIM(REPLACE(long, ',', '.')) AS DOUBLE) BETWEEN 163 AND 169
    )
    SELECT 
        latitude,
        longitude,
        EXTRACT(HOUR FROM datetime_accident) AS hour,
        EXTRACT(DOW FROM datetime_accident) AS dayofweek,
        EXTRACT(MONTH FROM datetime_accident) AS month,
        EXTRACT(YEAR FROM datetime_accident) AS year,
        atm
    FROM caracteristiques
    WHERE datetime_accident IS NOT NULL
    ORDER BY datetime_accident
    """
    
    try:
        accidents = conn.execute(query).df()
        conn.close()
        
        print(f"✅ {len(accidents)} accidents chargés")
        print(f"📅 Période: {accidents['year'].min():.0f} - {accidents['year'].max():.0f}")
        
        return accidents
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        conn.close()
        raise


def load_road_network():
    """Charge le réseau routier depuis routes.nc ou fichier pkl"""
    print("\n🗺️  Chargement du réseau routier...")
    
    if os.path.exists('routes_grid.pkl'):
        print("   Utilisation de routes_grid.pkl (grille pré-calculée)")
        routes_grid = joblib.load('routes_grid.pkl')
        print(f"✅ {len(routes_grid)} points de grille chargés")
        return routes_grid
    
    elif os.path.exists('routes.nc'):
        print("   Utilisation de routes.nc (GeoJSON) - Génération de la grille...")
        routes_gdf = gpd.read_file('routes.nc')
        
        # Créer une grille spatiale
        step = 0.02  # ~2km
        buffer_meters = 200
        lat_min, lat_max = -23.0, -19.5
        lon_min, lon_max = 163.5, 168.0
        
        lats = np.arange(lat_min, lat_max, step)
        lons = np.arange(lon_min, lon_max, step)
        grid = pd.DataFrame([(lat, lon) for lat in lats for lon in lons], columns=["latitude", "longitude"])
        grid["geometry"] = grid.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
        grid_gdf = gpd.GeoDataFrame(grid, geometry="geometry", crs="EPSG:4326").to_crs(epsg=3857)
        
        # Buffer autour des routes
        routes_buffer = routes_gdf.to_crs(epsg=3857).buffer(buffer_meters)
        routes_buffer_gdf = gpd.GeoDataFrame(geometry=routes_buffer, crs="EPSG:3857")
        
        # Spatial join
        grid_on_roads = gpd.sjoin(grid_gdf, routes_buffer_gdf, how="inner", predicate="intersects").drop(columns="index_right", errors='ignore')
        routes_grid = grid_on_roads.to_crs(epsg=4326)
        
        # Extraire les features OSM
        print("   Extraction des caractéristiques OSM...")
        road_types_mapping = {
            'motorway': 5, 'trunk': 4, 'primary': 3,
            'secondary': 2, 'tertiary': 2, 'residential': 1,
            'unclassified': 1, 'service': 1
        }
        
        def get_road_type(highway):
            if isinstance(highway, (np.ndarray, list)):
                highway = highway[0] if len(highway) > 0 else 'unclassified'
            highway = str(highway) if highway is not None else 'unclassified'
            return road_types_mapping.get(highway, 1)
        
        # Trouver la route la plus proche pour chaque point de grille
        from scipy.spatial.distance import cdist
        
        full_coords = routes_grid[['latitude', 'longitude']].values
        route_centroids = routes_gdf.geometry.centroid
        route_coords = np.array([[p.y, p.x] for p in route_centroids])
        
        # Par batch pour économiser la mémoire
        batch_size = 1000
        road_types = []
        speed_limits = []
        
        for i in range(0, len(full_coords), batch_size):
            batch = full_coords[i:i+batch_size]
            distances = cdist(batch, route_coords)
            nearest_indices = distances.argmin(axis=1)
            
            if 'highway' in routes_gdf.columns:
                road_types.extend([get_road_type(routes_gdf.iloc[idx]['highway']) for idx in nearest_indices])
            else:
                road_types.extend([2] * len(nearest_indices))
            
            if 'maxspeed' in routes_gdf.columns:
                for idx in nearest_indices:
                    speed = routes_gdf.iloc[idx].get('maxspeed', 50)
                    # Gérer les arrays/lists
                    if isinstance(speed, (np.ndarray, list)):
                        speed = speed[0] if len(speed) > 0 else 50
                    # Gérer les NaN
                    if speed is None or (isinstance(speed, float) and np.isnan(speed)):
                        speed = 50
                    # Parser les strings
                    elif isinstance(speed, str):
                        try:
                            speed = int(speed.split()[0]) if speed.split()[0].isdigit() else 50
                        except:
                            speed = 50
                    # Sinon, utiliser tel quel ou défaut
                    else:
                        try:
                            speed = int(speed)
                        except:
                            speed = 50
                    speed_limits.append(speed)
            else:
                speed_limits.extend([50] * len(nearest_indices))
        
        routes_grid['road_type'] = road_types
        routes_grid['speed_limit'] = speed_limits
        routes_grid = routes_grid[['latitude', 'longitude', 'road_type', 'speed_limit']].copy()
        
        # Sauvegarder pour éviter de recalculer
        joblib.dump(routes_grid, 'routes_grid.pkl')
        print(f"✅ {len(routes_grid)} points de grille chargés et sauvegardés")
        return routes_grid
        
    else:
        raise FileNotFoundError(
            "❌ Fichier réseau routier introuvable.\n"
            "   Exécutez d'abord: marimo run accident_fetch_data.py"
        )


def compute_accident_density(routes_grid, accidents, radius_km=5.0):
    """
    Calcule la densité d'accidents historiques dans un rayon donné pour chaque point.
    
    Args:
        routes_grid: DataFrame avec colonnes latitude, longitude
        accidents: DataFrame des accidents historiques
        radius_km: Rayon de recherche en kilomètres
    
    Returns:
        routes_grid avec colonne accident_density_5km ajoutée
    """
    print(f"\n🔢 Calcul de la densité d'accidents (rayon {radius_km}km)...")
    
    accident_coords = accidents[['latitude', 'longitude']].values
    grid_coords = routes_grid[['latitude', 'longitude']].values
    
    radius_deg = radius_km / 111.0
    
    print("   Création de l'index spatial...")
    nn = NearestNeighbors(radius=radius_deg, metric='haversine', algorithm='ball_tree')
    nn.fit(np.radians(accident_coords))
    
    print("   Comptage des accidents par zone...")
    distances, indices = nn.radius_neighbors(np.radians(grid_coords))
    
    density = np.array([len(idx) for idx in indices])
    
    routes_grid['accident_density_5km'] = density
    
    print(f"✅ Densité calculée:")
    print(f"   - Minimum: {density.min()} accidents")
    print(f"   - Maximum: {density.max()} accidents")
    print(f"   - Moyenne: {density.mean():.2f} accidents")
    print(f"   - Médiane: {np.median(density):.1f} accidents")
    
    high_density_zones = (density >= np.percentile(density, 90)).sum()
    print(f"   - Zones haute densité (top 10%): {high_density_zones}")
    
    return routes_grid


def compute_distance_to_noumea(routes_grid):
    """Calcule la distance de chaque point à Nouméa"""
    print("\n📍 Calcul de la distance à Nouméa...")
    
    noumea_center = (-22.2758, 166.4580)
    
    distances = routes_grid.apply(
        lambda row: geodesic(
            (row['latitude'], row['longitude']),
            noumea_center
        ).km,
        axis=1
    )
    
    routes_grid['dist_to_noumea_km'] = distances
    
    print(f"✅ Distances calculées:")
    print(f"   - Minimum: {distances.min():.1f} km")
    print(f"   - Maximum: {distances.max():.1f} km")
    print(f"   - Moyenne: {distances.mean():.1f} km")
    
    return routes_grid


def compute_temporal_accident_patterns(accidents):
    """Calcule les patterns temporels moyens des accidents"""
    print("\n⏰ Analyse des patterns temporels...")
    
    hourly_avg = accidents.groupby('hour').size() / len(accidents['year'].unique())
    dayofweek_avg = accidents.groupby('dayofweek').size() / len(accidents['year'].unique())
    monthly_avg = accidents.groupby('month').size() / len(accidents['year'].unique())
    
    patterns = {
        'hourly_risk': hourly_avg.to_dict(),
        'dayofweek_risk': dayofweek_avg.to_dict(),
        'monthly_risk': monthly_avg.to_dict()
    }
    
    print(f"✅ Patterns calculés:")
    print(f"   - Heure la plus dangereuse: {hourly_avg.idxmax()}h ({hourly_avg.max():.1f} accidents/an)")
    print(f"   - Jour le plus dangereux: {dayofweek_avg.idxmax()} ({dayofweek_avg.max():.1f} accidents/an)")
    print(f"   - Mois le plus dangereux: {monthly_avg.idxmax()} ({monthly_avg.max():.1f} accidents/an)")
    
    return patterns


def save_enriched_grid(routes_grid, patterns, output_file='routes_with_features.pkl'):
    """Sauvegarde la grille enrichie avec toutes les features pré-calculées"""
    print(f"\n💾 Sauvegarde des données enrichies dans {output_file}...")
    
    data = {
        'routes_grid': routes_grid,
        'temporal_patterns': patterns,
        'metadata': {
            'n_points': len(routes_grid),
            'density_computed': 'accident_density_5km' in routes_grid.columns,
            'distance_computed': 'dist_to_noumea_km' in routes_grid.columns,
            'features': list(routes_grid.columns)
        }
    }
    
    joblib.dump(data, output_file)
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✅ Fichier sauvegardé ({file_size_mb:.1f} MB)")
    print(f"   Features incluses: {', '.join(routes_grid.columns)}")


def main():
    """Pipeline principal de pré-calcul"""
    print("🚀 CALCUL DE DENSITÉ D'ACCIDENTS\n")
    
    try:
        accidents = load_accidents_data()
        
        routes_grid = load_road_network()
        
        routes_grid = compute_accident_density(routes_grid, accidents, radius_km=5.0)
        
        routes_grid = compute_distance_to_noumea(routes_grid)
        
        patterns = compute_temporal_accident_patterns(accidents)
        
        save_enriched_grid(routes_grid, patterns)
        
        print("\n" + "="*60)
        print("✅ PRÉ-CALCUL TERMINÉ AVEC SUCCÈS")
        print("="*60)
        print("\nProchaines étapes:")
        print("1. Mettre à jour predict_map.py pour charger routes_with_features.pkl")
        print("2. Mettre à jour predict_daily.py de la même manière")
        print("3. Re-entraîner le modèle avec: marimo run accident_fetch_data.py")
        
    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
