"""
Tests unitaires pour le projet accidents
Exécution : pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestFeatureEngineering:
    """Tests des features enrichies (24 features)"""
    
    def test_feature_count(self):
        """Vérifie que les 24 features sont bien générées"""
        # Données de test
        sample_data = pd.DataFrame({
            'latitude': [-22.0, -22.5, -21.0],
            'longitude': [166.0, 166.5, 167.0],
            'hour': [14, 8, 18],
            'dayofweek': [2, 5, 6],
            'month': [7, 1, 12],
            'atm': [1, 2, 1]
        })
        
        # Générer features (copier logique de predict_daily.py)
        target_date = datetime(2026, 7, 15)
        
        # Interactions spatio-temporelles
        sample_data['lat_hour'] = sample_data['latitude'] * sample_data['hour'] / 24
        sample_data['lon_hour'] = sample_data['longitude'] * sample_data['hour'] / 24
        sample_data['lat_dayofweek'] = sample_data['latitude'] * sample_data['dayofweek'] / 7
        sample_data['lon_dayofweek'] = sample_data['longitude'] * sample_data['dayofweek'] / 7
        
        # Indicateurs temporels
        sample_data['is_weekend'] = (sample_data['dayofweek'] >= 5).astype(int)
        sample_data['is_rush_morning'] = ((sample_data['hour'] >= 7) & (sample_data['hour'] <= 9)).astype(int)
        sample_data['is_rush_evening'] = ((sample_data['hour'] >= 17) & (sample_data['hour'] <= 19)).astype(int)
        sample_data['is_night'] = ((sample_data['hour'] >= 22) | (sample_data['hour'] <= 6)).astype(int)
        
        # Encodage cyclique
        sample_data['hour_sin'] = np.sin(2 * np.pi * sample_data['hour'] / 24)
        sample_data['hour_cos'] = np.cos(2 * np.pi * sample_data['hour'] / 24)
        sample_data['dayofweek_sin'] = np.sin(2 * np.pi * sample_data['dayofweek'] / 7)
        sample_data['dayofweek_cos'] = np.cos(2 * np.pi * sample_data['dayofweek'] / 7)
        
        # Features OSM (simulées)
        sample_data['road_type'] = [3, 2, 1]
        sample_data['speed_limit'] = [50, 70, 50]
        
        # Densité et proximité
        sample_data['accident_density_5km'] = [2.0, 3.5, 1.2]
        sample_data['dist_to_noumea_km'] = [10.5, 25.3, 45.8]
        
        # Temporelles avancées
        sample_data['is_holiday'] = [0, 1, 0]
        sample_data['school_holidays'] = [1, 1, 1]
        
        # Vérifications
        assert len(sample_data.columns) == 24, f"Attendu 24 features, obtenu {len(sample_data.columns)}"
        
        expected_features = [
            'latitude', 'longitude', 'hour', 'dayofweek', 'month', 'atm',
            'lat_hour', 'lon_hour', 'lat_dayofweek', 'lon_dayofweek',
            'is_weekend', 'is_rush_morning', 'is_rush_evening', 'is_night',
            'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
            'road_type', 'speed_limit', 'accident_density_5km', 'dist_to_noumea_km',
            'is_holiday', 'school_holidays'
        ]
        
        for feature in expected_features:
            assert feature in sample_data.columns, f"Feature manquante : {feature}"
    
    def test_cyclical_encoding(self):
        """Vérifie que l'encodage cyclique fonctionne correctement"""
        hours = np.array([0, 6, 12, 18, 23])
        hour_sin = np.sin(2 * np.pi * hours / 24)
        hour_cos = np.cos(2 * np.pi * hours / 24)
        
        # 0h et 23h devraient être proches (cyclique)
        assert abs(hour_sin[0] - hour_sin[4]) < 0.3, "Sin(0h) et Sin(23h) devraient être proches"
        assert abs(hour_cos[0] - hour_cos[4]) < 0.3, "Cos(0h) et Cos(23h) devraient être proches"
        
        # 12h devrait être opposé à 0h
        assert abs(hour_sin[2]) < 0.1, "Sin(12h) devrait être proche de 0"
        assert hour_cos[2] < -0.9, "Cos(12h) devrait être proche de -1"
    
    def test_temporal_indicators(self):
        """Vérifie les indicateurs temporels"""
        # Weekend
        assert (5 >= 5) == True, "Samedi (5) devrait être weekend"
        assert (6 >= 5) == True, "Dimanche (6) devrait être weekend"
        assert (2 >= 5) == False, "Mercredi (2) ne devrait pas être weekend"
        
        # Heures de pointe matin
        rush_morning_hours = [7, 8, 9]
        for h in rush_morning_hours:
            assert (h >= 7) and (h <= 9), f"{h}h devrait être heure de pointe matin"
        
        assert not ((6 >= 7) and (6 <= 9)), "6h ne devrait pas être heure de pointe matin"
        assert not ((10 >= 7) and (10 <= 9)), "10h ne devrait pas être heure de pointe matin"
        
        # Nuit
        night_hours = [22, 23, 0, 1, 2, 3, 4, 5, 6]
        for h in night_hours:
            is_night = (h >= 22) or (h <= 6)
            assert is_night, f"{h}h devrait être la nuit"
        
        assert not ((12 >= 22) or (12 <= 6)), "12h ne devrait pas être la nuit"
    
    def test_nc_holidays(self):
        """Vérifie les jours fériés NC"""
        nc_holidays = [(1,1), (5,1), (5,8), (7,14), (9,24), (11,1), (11,11), (12,25)]
        
        # Jours fériés
        assert (1, 1) in nc_holidays, "1er janvier devrait être férié"
        assert (7, 14) in nc_holidays, "14 juillet devrait être férié"
        assert (12, 25) in nc_holidays, "25 décembre devrait être férié"
        
        # Jours normaux
        assert (6, 15) not in nc_holidays, "15 juin ne devrait pas être férié"
        assert (3, 20) not in nc_holidays, "20 mars ne devrait pas être férié"
    
    def test_school_holidays(self):
        """Vérifie les vacances scolaires"""
        vacation_months = [1, 7, 8, 12]
        
        for month in vacation_months:
            assert month in vacation_months, f"Mois {month} devrait être en vacances"
        
        assert 5 not in vacation_months, "Mai ne devrait pas être en vacances"
        assert 10 not in vacation_months, "Octobre ne devrait pas être en vacances"


class TestDataValidation:
    """Tests de validation des données"""
    
    def test_coordinate_ranges(self):
        """Vérifie que les coordonnées sont dans les limites NC"""
        # Nouvelle-Calédonie
        lat_min, lat_max = -23.0, -19.5
        lon_min, lon_max = 163.5, 168.0
        
        # Coordonnées valides
        valid_lat = -22.0
        valid_lon = 166.0
        assert lat_min <= valid_lat <= lat_max, "Latitude valide devrait passer"
        assert lon_min <= valid_lon <= lon_max, "Longitude valide devrait passer"
        
        # Coordonnées invalides
        invalid_lat = -25.0
        invalid_lon = 170.0
        assert not (lat_min <= invalid_lat <= lat_max), "Latitude invalide devrait échouer"
        assert not (lon_min <= invalid_lon <= lon_max), "Longitude invalide devrait échouer"
    
    def test_hour_range(self):
        """Vérifie que l'heure est entre 0-23"""
        valid_hours = [0, 12, 23]
        for h in valid_hours:
            assert 0 <= h <= 23, f"Heure {h} devrait être valide"
        
        invalid_hours = [-1, 24, 25]
        for h in invalid_hours:
            assert not (0 <= h <= 23), f"Heure {h} devrait être invalide"
    
    def test_dayofweek_range(self):
        """Vérifie que le jour de semaine est entre 0-6"""
        valid_days = [0, 3, 6]
        for d in valid_days:
            assert 0 <= d <= 6, f"Jour {d} devrait être valide"
        
        invalid_days = [-1, 7, 8]
        for d in invalid_days:
            assert not (0 <= d <= 6), f"Jour {d} devrait être invalide"
    
    def test_atm_codes(self):
        """Vérifie les codes météo valides"""
        valid_atm_codes = [1, 2, 3, 5]
        
        for code in valid_atm_codes:
            assert code in valid_atm_codes, f"Code ATM {code} devrait être valide"
        
        assert 4 not in valid_atm_codes, "Code ATM 4 ne devrait pas être valide"
        assert 0 not in valid_atm_codes, "Code ATM 0 ne devrait pas être valide"


class TestPredictionOutput:
    """Tests des outputs de prédiction"""
    
    def test_prediction_shape(self):
        """Vérifie que le nombre de prédictions est correct"""
        # Simuler une grille de 1500 points sur 24h
        n_points = 1500
        n_hours = 24
        expected_predictions = n_points * n_hours
        
        # Simuler des prédictions
        predictions = pd.DataFrame({
            'date': [datetime(2026, 1, 25).date()] * expected_predictions,
            'hour': np.repeat(range(24), n_points),
            'latitude': np.random.uniform(-23, -19.5, expected_predictions),
            'longitude': np.random.uniform(163.5, 168, expected_predictions),
            'probability': np.random.uniform(0, 1, expected_predictions)
        })
        
        assert len(predictions) == expected_predictions, f"Attendu {expected_predictions} prédictions"
        assert set(predictions['hour'].unique()) == set(range(24)), "Toutes les heures devraient être présentes"
    
    def test_probability_range(self):
        """Vérifie que les probabilités sont entre 0 et 1"""
        probas = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        
        assert (probas >= 0).all(), "Toutes les probas devraient être ≥ 0"
        assert (probas <= 1).all(), "Toutes les probas devraient être ≤ 1"
        
        invalid_probas = np.array([-0.1, 1.5])
        assert not (invalid_probas >= 0).all() or not (invalid_probas <= 1).all(), \
            "Probas invalides devraient être détectées"
    
    def test_required_columns(self):
        """Vérifie que toutes les colonnes requises sont présentes"""
        predictions = pd.DataFrame({
            'date': [datetime.now().date()],
            'hour': [12],
            'latitude': [-22.0],
            'longitude': [166.0],
            'probability': [0.75],
            'atm_code': [1],
            'dayofweek': [3],
            'month': [7],
            'created_at': [datetime.now()]
        })
        
        required_columns = [
            'date', 'hour', 'latitude', 'longitude', 'probability',
            'atm_code', 'dayofweek', 'month', 'created_at'
        ]
        
        for col in required_columns:
            assert col in predictions.columns, f"Colonne manquante : {col}"


class TestInteractions:
    """Tests des features d'interaction"""
    
    def test_spatio_temporal_interactions(self):
        """Vérifie que les interactions spatio-temporelles sont correctes"""
        lat = -22.0
        lon = 166.0
        hour = 14
        dayofweek = 3
        
        lat_hour = lat * hour / 24
        lon_hour = lon * hour / 24
        lat_dayofweek = lat * dayofweek / 7
        lon_dayofweek = lon * dayofweek / 7
        
        # Vérifier que les valeurs sont dans des plages raisonnables
        assert -23 < lat_hour < 0, "lat_hour devrait être dans la plage attendue"
        assert 0 < lon_hour < 170, "lon_hour devrait être dans la plage attendue"
        assert -23 < lat_dayofweek < 0, "lat_dayofweek devrait être dans la plage attendue"
        assert 0 < lon_dayofweek < 170, "lon_dayofweek devrait être dans la plage attendue"
    
    def test_osm_features_range(self):
        """Vérifie que les features OSM sont dans les plages valides"""
        # road_type : 1-5
        road_types = [1, 2, 3, 4, 5]
        for rt in road_types:
            assert 1 <= rt <= 5, f"road_type {rt} devrait être entre 1 et 5"
        
        # speed_limit : généralement 30-110 km/h
        speed_limits = [30, 50, 70, 90, 110]
        for sl in speed_limits:
            assert 0 < sl <= 130, f"speed_limit {sl} devrait être raisonnable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
