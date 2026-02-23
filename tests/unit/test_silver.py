"""
Tests unitaires pour la couche Silver (features)
"""

import pytest
import polars as pl
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch
from src.accidents.silver.features import (
    generate_negative_samples,
    add_osm_features,
    create_full_dataset
)
from src.accidents.utils.temporal import add_temporal_features


class TestTemporalFeatures:
    """Tests pour les features temporelles"""
    
    def test_add_temporal_features_creates_columns(self):
        """Vérifie que toutes les features temporelles sont créées"""
        # Données de test
        df = pl.DataFrame({
            'Num_Acc': ['202400001', '202400002'],
            'accident_datetime': [
                datetime(2024, 7, 15, 14, 30),  # Lundi après-midi
                datetime(2024, 7, 20, 8, 0),    # Samedi matin
            ],
        })
        
        result = add_temporal_features(df)
        
        # Vérifier les colonnes créées
        expected_cols = [
            'hour_of_day', 'day_of_week', 'month',
            'hour_sin', 'hour_cos',
            'day_sin', 'day_cos',
            'month_sin', 'month_cos',
            'is_weekend', 'is_rush_hour', 'is_holiday'
        ]
        
        for col in expected_cols:
            assert col in result.columns, f"Colonne {col} manquante"
    
    def test_weekend_detection(self):
        """Vérifie la détection weekend"""
        df = pl.DataFrame({
            'accident_datetime': [
                datetime(2024, 7, 15, 10, 0),  # Lundi
                datetime(2024, 7, 20, 10, 0),  # Samedi
                datetime(2024, 7, 21, 10, 0),  # Dimanche
            ],
        })
        
        result = add_temporal_features(df)
        
        # Lundi = 0, Samedi = 1, Dimanche = 1
        assert result['is_weekend'][0] == 0
        assert result['is_weekend'][1] == 1
        assert result['is_weekend'][2] == 1
    
    def test_rush_hour_detection(self):
        """Vérifie la détection heures de pointe"""
        df = pl.DataFrame({
            'accident_datetime': [
                datetime(2024, 7, 15, 8, 0),   # Rush matin
                datetime(2024, 7, 15, 12, 0),  # Pas rush
                datetime(2024, 7, 15, 18, 0),  # Rush soir
            ],
        })
        
        result = add_temporal_features(df)
        
        # Rush = 7-9h ou 17-19h
        assert result['is_rush_hour'][0] == 1
        assert result['is_rush_hour'][1] == 0
        assert result['is_rush_hour'][2] == 1
    
    def test_cyclical_encoding(self):
        """Vérifie l'encodage cyclique (sin/cos)"""
        df = pl.DataFrame({
            'accident_datetime': [
                datetime(2024, 7, 15, 0, 0),   # Minuit
                datetime(2024, 7, 15, 12, 0),  # Midi
            ],
        })
        
        result = add_temporal_features(df)
        
        # Minuit : hour_sin ≈ 0, hour_cos ≈ 1
        # Midi : hour_sin ≈ 0, hour_cos ≈ -1
        assert abs(result['hour_sin'][0]) < 0.1
        assert result['hour_cos'][0] > 0.9
        assert abs(result['hour_sin'][1]) < 0.1
        assert result['hour_cos'][1] < -0.9


class TestNegativeSampling:
    """Tests pour la génération d'échantillons négatifs"""
    
    def test_generate_negative_samples_count(self):
        """Vérifie que le bon nombre d'échantillons est généré"""
        # Données positives
        positives = pl.DataFrame({
            'latitude': [-22.27, -22.30],
            'longitude': [166.45, 166.50],
            'accident_datetime': [
                datetime(2024, 7, 15, 14, 0),
                datetime(2024, 7, 20, 8, 0),
            ],
        })
        
        # Grille de points
        grid = pl.DataFrame({
            'latitude': [-22.26, -22.28, -22.29, -22.31],
            'longitude': [166.44, 166.46, 166.48, 166.51],
        })
        
        result = generate_negative_samples(positives, grid, ratio=2.0)
        
        # 2 positives × 2.0 = 4 négatives attendues
        assert len(result) == 4
        assert 'target' in result.columns
        assert result['target'].unique().to_list() == [0]
    
    def test_negative_samples_avoid_accidents(self):
        """Vérifie que les négatifs évitent les accidents (300m)"""
        positives = pl.DataFrame({
            'latitude': [-22.27],
            'longitude': [166.45],
            'accident_datetime': [datetime(2024, 7, 15, 14, 0)],
        })
        
        # Grille avec point très proche (< 300m) et point loin
        grid = pl.DataFrame({
            'latitude': [-22.27001, -22.30],  # ~11m vs ~3.3km
            'longitude': [166.45001, 166.50],
        })
        
        result = generate_negative_samples(positives, grid, ratio=1.0, exclusion_radius_m=300)
        
        # Le point à 11m devrait être exclu, seul le point à 3.3km reste
        assert len(result) == 1
        assert abs(result['latitude'][0] - (-22.30)) < 0.01


class TestOSMFeatures:
    """Tests pour les features OSM"""
    
    @pytest.mark.integration
    def test_add_osm_features_mock(self):
        """Test avec données OSM mockées"""
        df = pl.DataFrame({
            'latitude': [-22.27, -22.30],
            'longitude': [166.45, 166.50],
        })
        
        # Mock OSM data
        mock_osm = pd.DataFrame({
            'latitude': [-22.27, -22.30],
            'longitude': [166.45, 166.50],
            'road_length': [150.0, 200.0],
            'road_count': [3, 5],
            'nearest_road_type': ['secondary', 'primary'],
        })
        
        with patch('src.accidents.silver.features.load_osm_data', return_value=mock_osm):
            result = add_osm_features(df)
        
        assert 'road_length' in result.columns
        assert 'road_count' in result.columns
        assert len(result) == 2


class TestFullDataset:
    """Tests pour la création du dataset complet"""
    
    def test_create_full_dataset_structure(self):
        """Vérifie la structure du dataset final"""
        mock_client = Mock()
        
        # Mock données accidents_nc
        accidents = pl.DataFrame({
            'Num_Acc': ['202400001'],
            'latitude': [-22.27],
            'longitude': [166.45],
            'accident_datetime': [datetime(2024, 7, 15, 14, 0)],
            'atm': ['1'],
        })
        
        mock_client.conn.execute.return_value.pl.return_value = accidents
        
        with patch('src.accidents.silver.features.get_client', return_value=mock_client):
            with patch('src.accidents.silver.features.generate_negative_samples') as mock_neg:
                with patch('src.accidents.silver.features.add_osm_features') as mock_osm:
                    # Mock retours
                    mock_neg.return_value = accidents.with_columns(pl.lit(0).alias('target'))
                    mock_osm.return_value = accidents
                    
                    result = create_full_dataset()
        
        assert isinstance(result, dict)
        assert 'rows' in result
        assert 'positives' in result
        assert 'negatives' in result


class TestDataQuality:
    """Tests de qualité des données silver"""
    
    def test_no_missing_coordinates(self):
        """Vérifie l'absence de coordonnées manquantes"""
        df = pl.DataFrame({
            'latitude': [-22.27, None, -22.30],
            'longitude': [166.45, 166.50, None],
        })
        
        # Filtrer les lignes avec coordonnées valides
        clean = df.filter(
            pl.col('latitude').is_not_null() & pl.col('longitude').is_not_null()
        )
        
        assert len(clean) == 1  # Seule la première ligne est complète
    
    def test_target_balance(self):
        """Vérifie l'équilibre des classes (target)"""
        df = pl.DataFrame({
            'target': [1, 1, 0, 0, 0]
        })
        
        counts = df.group_by('target').agg(pl.count().alias('n'))
        
        # Ratio négatifs/positifs
        n_positives = counts.filter(pl.col('target') == 1)['n'][0]
        n_negatives = counts.filter(pl.col('target') == 0)['n'][0]
        ratio = n_negatives / n_positives
        
        # On attend un ratio ~2-5
        assert 1.0 <= ratio <= 6.0
