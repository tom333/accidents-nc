#!/usr/bin/env python3
"""
Script de prédiction quotidienne des accidents
Génère les prédictions pour le lendemain et les stocke dans DuckDB
Usage: python predict_daily.py [--date YYYY-MM-DD] [--atm CODE]
"""

import argparse
import joblib
import pandas as pd
import numpy as np
import geopandas as gpd
import duckdb
from datetime import datetime, timedelta
from pathlib import Path
from shapely.geometry import Point


def load_model_and_data():
    """Charge le modèle et la grille routière"""
    print("📦 Chargement du modèle...")
    model = joblib.load('accident_model.pkl')
    encoder = joblib.load('atm_encoder.pkl')
    features = joblib.load('features.pkl')
    
    print("🗺️  Chargement de la grille routière...")
    routes_osm = gpd.read_file('routes.nc')
    
    # Recréer la grille spatiale (même logique que predict_map.py)
    grid_step = 0.02
    buffer_meters = 200
    lat_min, lat_max = -23.0, -19.5
    lon_min, lon_max = 163.5, 168.0
    
    lats = np.arange(lat_min, lat_max, grid_step)
    lons = np.arange(lon_min, lon_max, grid_step)
    
    grid = pd.DataFrame([(lat, lon) for lat in lats for lon in lons], columns=["latitude", "longitude"])
    grid["geometry"] = grid.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
    grid_gdf = gpd.GeoDataFrame(grid, geometry="geometry", crs="EPSG:4326").to_crs(epsg=3857)
    
    routes_buffer = routes_osm.to_crs(epsg=3857).buffer(buffer_meters)
    routes_buffer_gdf = gpd.GeoDataFrame(geometry=routes_buffer, crs="EPSG:3857")
    
    grid_on_roads = gpd.sjoin(grid_gdf, routes_buffer_gdf, how="inner", predicate="intersects").drop(columns="index_right")
    routes_grid = grid_on_roads.to_crs(epsg=4326).drop_duplicates(subset=['latitude', 'longitude'])
    
    print(f"✅ Grille prête : {len(routes_grid)} points")
    
    return model, encoder, features, routes_grid


def generate_predictions(model, features, routes_grid, target_date, atm_code=1):
    """
    Génère les prédictions pour toutes les heures d'une journée
    
    Args:
        model: Modèle ML entraîné
        features: Liste des features
        routes_grid: GeoDataFrame de la grille
        target_date: datetime de la date cible
        atm_code: Code conditions météo (1=Normal, 2=Pluie légère, 3=Pluie forte, 5=Brouillard)
    
    Returns:
        DataFrame avec les prédictions
    """
    print(f"🔮 Génération des prédictions pour {target_date.strftime('%Y-%m-%d')}...")
    
    # Extraire coordonnées
    grid_lats = routes_grid.geometry.y.values
    grid_lons = routes_grid.geometry.x.values
    
    all_predictions = []
    
    for hour in range(24):
        # Créer DataFrame pour cette heure
        hourly_data = pd.DataFrame({
            'latitude': grid_lats,
            'longitude': grid_lons,
            'hour': hour,
            'dayofweek': target_date.weekday(),
            'month': target_date.month,
            'atm': atm_code
        })
        
        # Prédictions
        probas = model.predict_proba(hourly_data[features])[:, 1]
        
        # Stocker toutes les prédictions (pas de filtrage par seuil)
        predictions = pd.DataFrame({
            'date': target_date.date(),
            'hour': hour,
            'latitude': hourly_data['latitude'].values,
            'longitude': hourly_data['longitude'].values,
            'probability': probas,
            'atm_code': atm_code,
            'dayofweek': target_date.weekday(),
            'month': target_date.month,
            'created_at': datetime.now()
        })
        
        all_predictions.append(predictions)
    
    # Concaténer toutes les heures
    result = pd.concat(all_predictions, ignore_index=True)
    
    print(f"✅ {len(result):,} prédictions générées")
    return result


def save_to_duckdb(predictions_df, db_path='predictions.duckdb'):
    """
    Sauvegarde les prédictions dans DuckDB
    
    Args:
        predictions_df: DataFrame des prédictions
        db_path: Chemin vers la base DuckDB
    """
    print(f"💾 Sauvegarde dans {db_path}...")
    
    con = duckdb.connect(db_path)
    
    # Créer la table si elle n'existe pas
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY,
            date DATE NOT NULL,
            hour INTEGER NOT NULL,
            latitude DOUBLE NOT NULL,
            longitude DOUBLE NOT NULL,
            probability DOUBLE NOT NULL,
            atm_code INTEGER NOT NULL,
            dayofweek INTEGER NOT NULL,
            month INTEGER NOT NULL,
            created_at TIMESTAMP NOT NULL,
            UNIQUE(date, hour, latitude, longitude)
        )
    """)
    
    # Créer un index pour les requêtes rapides
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_date_hour 
        ON predictions(date, hour)
    """)
    
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_probability 
        ON predictions(probability)
    """)
    
    # Insérer les données (remplacer si existe déjà)
    try:
        con.execute("DELETE FROM predictions WHERE date = ?", [predictions_df['date'].iloc[0]])
        con.execute("INSERT INTO predictions SELECT ROW_NUMBER() OVER () as id, * FROM predictions_df")
        con.commit()
        
        # Statistiques
        stats = con.execute("""
            SELECT 
                COUNT(*) as total_points,
                AVG(probability) as avg_probability,
                MAX(probability) as max_probability,
                COUNT(CASE WHEN probability >= 0.5 THEN 1 END) as high_risk_points
            FROM predictions 
            WHERE date = ?
        """, [predictions_df['date'].iloc[0]]).fetchone()
        
        print(f"✅ Données sauvegardées :")
        print(f"   • Total points : {stats[0]:,}")
        print(f"   • Probabilité moyenne : {stats[1]:.2%}")
        print(f"   • Probabilité maximale : {stats[2]:.2%}")
        print(f"   • Points à risque élevé (≥50%) : {stats[3]:,}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde : {e}")
        con.rollback()
        raise
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(
        description="Génère les prédictions d'accidents pour le lendemain"
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Date cible (YYYY-MM-DD). Par défaut : demain'
    )
    parser.add_argument(
        '--atm',
        type=int,
        default=1,
        choices=[1, 2, 3, 5],
        help='Code conditions météo (1=Normal, 2=Pluie légère, 3=Pluie forte, 5=Brouillard)'
    )
    parser.add_argument(
        '--db',
        type=str,
        default='predictions.duckdb',
        help='Chemin vers la base DuckDB'
    )
    
    args = parser.parse_args()
    
    # Date cible
    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d')
    else:
        target_date = datetime.now() + timedelta(days=1)
    
    print(f"🚀 Prédiction pour {target_date.strftime('%Y-%m-%d')}")
    print(f"🌦️  Conditions météo : {args.atm}")
    print()
    
    # Pipeline
    try:
        model, encoder, features, routes_grid = load_model_and_data()
        predictions = generate_predictions(model, features, routes_grid, target_date, args.atm)
        save_to_duckdb(predictions, args.db)
        
        print()
        print("✅ Prédictions générées avec succès !")
        print(f"📊 Consultez la base : duckdb {args.db}")
        
    except FileNotFoundError as e:
        print(f"❌ Fichier manquant : {e}")
        print("💡 Assurez-vous d'avoir exécuté accident_fetch_data.py d'abord")
        return 1
    except Exception as e:
        print(f"❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
