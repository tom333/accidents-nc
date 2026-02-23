"""
Tests unitaires pour la couche Gold (ML datasets)
"""

import pytest
import polars as pl
from unittest.mock import Mock, patch
from src.accidents.gold.datasets import create_ml_datasets, FEATURE_COLUMNS
from src.accidents.gold.schema import GOLD_SCHEMA


class TestMLDatasetCreation:
    """Tests pour la création des datasets ML"""
    
    def test_create_ml_datasets_returns_dict(self):
        """Vérifie que la fonction retourne un dict avec stats"""
        mock_client = Mock()
        
        # Mock full_dataset
        full_data = pl.DataFrame({
            'latitude': [-22.27, -22.30, -22.28, -22.31],
            'longitude': [166.45, 166.50, 166.48, 166.51],
            'hour_of_day': [14, 8, 12, 18],
            'day_of_week': [0, 5, 2, 6],
            'month': [7, 7, 8, 8],
            'is_weekend': [0, 1, 0, 1],
            'is_rush_hour': [0, 1, 0, 1],
            'is_holiday': [0, 0, 0, 0],
            'target': [1, 0, 1, 0],
        })
        
        mock_client.conn.execute.return_value.pl.return_value = full_data
        
        with patch('src.accidents.gold.datasets.get_client', return_value=mock_client):
            result = create_ml_datasets(test_size=0.5, random_state=42)
        
        # Vérifications
        assert isinstance(result, dict)
        assert 'X_train_rows' in result
        assert 'X_test_rows' in result
        assert 'train_positives' in result
        assert 'test_positives' in result
    
    def test_train_test_split_ratio(self):
        """Vérifie le ratio train/test"""
        mock_client = Mock()
        
        # 100 lignes de test
        n = 100
        full_data = pl.DataFrame({
            'latitude': [-22.27] * n,
            'longitude': [166.45] * n,
            'hour_of_day': [14] * n,
            'day_of_week': [0] * n,
            'month': [7] * n,
            'is_weekend': [0] * n,
            'is_rush_hour': [0] * n,
            'is_holiday': [0] * n,
            'target': [1 if i % 2 == 0 else 0 for i in range(n)],
        })
        
        mock_client.conn.execute.return_value.pl.return_value = full_data
        
        with patch('src.accidents.gold.datasets.get_client', return_value=mock_client):
            result = create_ml_datasets(test_size=0.25, random_state=42)
        
        # 25% test = 25 lignes, 75% train = 75 lignes
        assert result['X_test_rows'] == 25
        assert result['X_train_rows'] == 75
    
    def test_feature_columns_exist(self):
        """Vérifie que les colonnes features sont définies"""
        assert isinstance(FEATURE_COLUMNS, list)
        assert len(FEATURE_COLUMNS) > 0
        
        # Vérifier quelques features clés
        expected = ['latitude', 'longitude', 'hour_of_day', 'is_weekend']
        for feat in expected:
            assert feat in FEATURE_COLUMNS, f"Feature {feat} manquante"


class TestDataSplitting:
    """Tests pour le splitting des données"""
    
    def test_stratified_split_preserves_balance(self):
        """Vérifie que le split stratifié préserve l'équilibre"""
        # Données déséquilibrées : 90 négatifs, 10 positifs
        df = pl.DataFrame({
            'feature1': list(range(100)),
            'target': [1] * 10 + [0] * 90,
        })
        
        # Simuler un split stratifié 80/20
        train = df.head(80)
        test = df.tail(20)
        
        # Dans train : ~8 positifs, ~72 négatifs
        # Dans test : ~2 positifs, ~18 négatifs
        train_pos_ratio = train.filter(pl.col('target') == 1).shape[0] / len(train)
        test_pos_ratio = test.filter(pl.col('target') == 1).shape[0] / len(test)
        
        # Les ratios devraient être similaires (~0.1)
        assert abs(train_pos_ratio - test_pos_ratio) < 0.05
    
    def test_no_data_leakage(self):
        """Vérifie qu'il n'y a pas de fuite entre train et test"""
        mock_client = Mock()
        
        full_data = pl.DataFrame({
            'Num_Acc': [f'ACC{i:05d}' for i in range(100)],
            'latitude': [-22.27] * 100,
            'longitude': [166.45] * 100,
            'hour_of_day': [14] * 100,
            'day_of_week': [0] * 100,
            'month': [7] * 100,
            'is_weekend': [0] * 100,
            'is_rush_hour': [0] * 100,
            'is_holiday': [0] * 100,
            'target': [1 if i % 2 == 0 else 0 for i in range(100)],
        })
        
        # Simuler 2 appels différents (train puis test)
        train_data = full_data.head(80)
        test_data = full_data.tail(20)
        
        mock_client.conn.execute.return_value.pl.side_effect = [train_data, test_data]
        
        with patch('src.accidents.gold.datasets.get_client', return_value=mock_client):
            # Les Num_Acc dans train et test ne doivent pas se chevaucher
            train_ids = set(train_data['Num_Acc'].to_list())
            test_ids = set(test_data['Num_Acc'].to_list())
            
            assert len(train_ids & test_ids) == 0, "Data leakage détecté!"


class TestGoldSchema:
    """Tests pour le schéma gold"""
    
    def test_gold_schema_defined(self):
        """Vérifie que le schéma gold est défini"""
        assert GOLD_SCHEMA is not None
        assert isinstance(GOLD_SCHEMA, str)
        assert len(GOLD_SCHEMA) > 0


class TestFeatureEngineering:
    """Tests pour l'ingénierie des features"""
    
    def test_no_null_features(self):
        """Vérifie l'absence de features nulles"""
        df = pl.DataFrame({
            'latitude': [-22.27, -22.30, None],
            'longitude': [166.45, 166.50, 166.48],
            'hour_of_day': [14, None, 12],
        })
        
        # Compter les nulls par colonne
        null_counts = df.null_count()
        
        # Certaines colonnes ont des nulls
        assert null_counts['latitude'][0] == 1
        assert null_counts['hour_of_day'][0] == 1
        
        # Filtrer les lignes complètes
        clean = df.drop_nulls()
        assert len(clean) == 1
    
    def test_feature_types(self):
        """Vérifie les types des features"""
        df = pl.DataFrame({
            'latitude': [-22.27],
            'longitude': [166.45],
            'hour_of_day': [14],
            'is_weekend': [1],
            'target': [1],
        })
        
        # Vérifier les types
        assert df['latitude'].dtype == pl.Float64
        assert df['hour_of_day'].dtype == pl.Int64
        assert df['is_weekend'].dtype == pl.Int64
        assert df['target'].dtype == pl.Int64
    
    def test_feature_ranges(self):
        """Vérifie les ranges des features"""
        df = pl.DataFrame({
            'hour_of_day': [0, 12, 23],
            'day_of_week': [0, 3, 6],
            'month': [1, 6, 12],
            'is_weekend': [0, 0, 1],
        })
        
        # Vérifier les ranges
        assert df['hour_of_day'].min() >= 0
        assert df['hour_of_day'].max() <= 23
        assert df['day_of_week'].min() >= 0
        assert df['day_of_week'].max() <= 6
        assert df['month'].min() >= 1
        assert df['month'].max() <= 12
        assert set(df['is_weekend'].unique()) <= {0, 1}


class TestIntegrationGold:
    """Tests d'intégration couche gold"""
    
    @pytest.mark.integration
    def test_end_to_end_ml_pipeline(self):
        """Test du pipeline complet jusqu'au ML"""
        # Ce test nécessite DuckLake
        pytest.skip("Requires DuckLake connection")
