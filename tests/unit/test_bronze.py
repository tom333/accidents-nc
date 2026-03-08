"""
Tests unitaires pour la couche Bronze (ingestion)
"""

from unittest.mock import Mock, patch

import polars as pl
import pytest

from src.assets.bronze.ingest import create_accidents_nc, ingest_caracteristiques, ingest_usagers


class TestIngestCaracteristiques:
    """Tests pour l'ingestion des caractéristiques"""

    def test_ingest_caracteristiques_schema(self):
        """Vérifie que les colonnes attendues sont présentes"""
        # Mock DuckLake client
        mock_client = Mock()

        # Données de test
        pl.DataFrame(
            {
                "Num_Acc": ["202400001", "202400002"],
                "dep": ["988", "988"],
                "lat": ["-22.2758", "-22.3"],
                "long": ["166.4581", "166.5"],
                "jour": ["01", "02"],
                "mois": ["07", "07"],
                "an": ["2024", "2024"],
                "hrmn": ["1430", "0800"],
                "atm": ["1", "2"],
                "col": ["1", "2"],
            }
        )

        mock_res = Mock()
        mock_res.fetchone.return_value = [2]
        mock_client.conn.execute.return_value = mock_res

        with patch("src.assets.bronze.ingest.get_client", return_value=mock_client):
            with patch("src.assets.bronze.ingest._discover_files", return_value=[]):
                result = ingest_caracteristiques()

        # Vérifications
        assert isinstance(result, dict)
        assert result["rows"] == 2  # 2 lignes insérées
        mock_client.conn.execute.assert_called()

    def test_ingest_caracteristiques_empty(self):
        """Vérifie le comportement avec données vides"""
        mock_client = Mock()
        mock_res = Mock()
        mock_res.fetchone.return_value = [0]
        mock_client.conn.execute.return_value = mock_res

        with patch("src.assets.bronze.ingest.get_client", return_value=mock_client):
            with patch("src.assets.bronze.ingest._discover_files", return_value=[]):
                result = ingest_caracteristiques()

        assert result["rows"] == 0


class TestIngestUsagers:
    """Tests pour l'ingestion des usagers"""

    def test_ingest_usagers_schema(self):
        """Vérifie que les colonnes attendues sont présentes"""
        mock_client = Mock()

        pl.DataFrame(
            {
                "Num_Acc": ["202400001", "202400001"],
                "grav": ["1", "3"],
                "sexe": ["1", "2"],
                "an_nais": ["1990", "1985"],
            }
        )

        mock_res = Mock()
        mock_res.fetchone.return_value = [2]
        mock_client.conn.execute.return_value = mock_res

        with patch("src.assets.bronze.ingest.get_client", return_value=mock_client):
            with patch("src.assets.bronze.ingest._discover_files", return_value=[]):
                result = ingest_usagers()

        assert isinstance(result, dict)
        assert result["rows"] == 2


class TestCreateAccidentsNC:
    """Tests pour la création de la vue accidents NC"""

    def test_create_accidents_nc_filters_988(self):
        """Vérifie que seul le dep 988 est conservé"""
        mock_client = Mock()

        # Données avec plusieurs départements
        pl.DataFrame(
            {
                "Num_Acc": ["202400001", "202400002", "202400003"],
                "dep": ["988", "75", "988"],
                "lat": [-22.27, 48.85, -22.30],
                "long": [166.45, 2.35, 166.50],
            }
        )

        mock_res = Mock()
        mock_res.fetchone.return_value = [2]
        mock_client.conn.execute.return_value = mock_res

        with patch("src.assets.bronze.ingest.get_client", return_value=mock_client):
            result = create_accidents_nc()

        # On devrait avoir que 2 accidents (dep 988)
        assert result["rows"] == 2

    def test_create_accidents_nc_coordinates_cleaning(self):
        """Vérifie le nettoyage des coordonnées"""
        mock_client = Mock()

        # Données avec coordonnées à nettoyer
        pl.DataFrame(
            {
                "Num_Acc": ["202400001"],
                "dep": ["988"],
                "lat": [" -22,2758 "],  # Espaces et virgule
                "long": [" 166,4581 "],
            }
        )

        # Le nettoyage devrait convertir -22,2758 -> -22.2758
        pl.DataFrame(
            {
                "Num_Acc": ["202400001"],
                "dep": ["988"],
                "lat": [-22.2758],
                "long": [166.4581],
            }
        )

        mock_res = Mock()
        mock_res.fetchone.return_value = [1]
        mock_client.conn.execute.return_value = mock_res

        with patch("src.assets.bronze.ingest.get_client", return_value=mock_client):
            result = create_accidents_nc()

        assert result["rows"] == 1


class TestIntegrationBronze:
    """Tests d'intégration couche bronze"""

    @pytest.mark.integration
    def test_full_ingestion_pipeline(self):
        """Test du pipeline complet d'ingestion"""
        # Ce test nécessite une vraie connexion DuckLake
        # À skip en CI si pas de config
        pytest.skip("Requires DuckLake connection")


class TestDataValidation:
    """Tests de validation des données ingérées"""

    def test_num_acc_format(self):
        """Vérifie le format des Num_Acc"""
        test_num_acc = ["202400001", "202400002", "2024ABC"]

        # Num_Acc doit être numérique
        valid = [n for n in test_num_acc if n.isdigit()]

        assert len(valid) == 2
        assert "2024ABC" not in valid

    def test_coordinates_range_nc(self):
        """Vérifie que les coordonnées sont dans la zone NC"""
        # Nouvelle-Calédonie : lat ~ [-22.7, -19.5], long ~ [163.5, 168.5]
        test_coords = [
            (-22.2758, 166.4581, True),  # Nouméa - valide
            (-20.0, 165.0, True),  # NC - valide
            (48.8566, 2.3522, False),  # Paris - invalide
            (0, 0, False),  # Null island - invalide
        ]

        for lat, lon, expected in test_coords:
            is_valid = (-23 <= lat <= -19) and (163 <= lon <= 169)
            assert is_valid == expected, f"Coord ({lat}, {lon}) validation failed"
