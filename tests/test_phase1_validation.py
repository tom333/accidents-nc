"""Test de validation Phase 1 - Structure et imports."""
import sys


def test_imports():
    """Vérifie que tous les imports fonctionnent."""
    print("🧪 Test 1/4 : Imports...")
    
    try:
        from src.accidents.ducklake import get_client
        from src.accidents.config import DuckLakeConfig
        print("   ✅ Core (ducklake, config)")
        
        from src.accidents.bronze.ingest import ingest_all
        from src.accidents.bronze.schema import BRONZE_SCHEMA
        print("   ✅ Bronze (ingest, schema)")
        
        from src.accidents.silver.features import build_feature_store
        from src.accidents.silver.schema import SILVER_SCHEMA
        print("   ✅ Silver (features, schema)")
        
        from src.accidents.gold.datasets import build_datasets
        from src.accidents.gold.schema import GOLD_SCHEMA
        print("   ✅ Gold (datasets, schema)")
        
        from src.accidents.utils import (
            buffer_routes, 
            create_spatial_grid,
            cyclical_encode,
            compute_classification_metrics
        )
        print("   ✅ Utils (spatial, temporal, metrics)")
        
        return True
    except ImportError as e:
        print(f"   ❌ Erreur import: {e}")
        return False


def test_no_dagster_dependency():
    """Vérifie que le code métier est indépendant de Dagster."""
    print("\n🧪 Test 2/4 : Indépendance Dagster...")
    
    try:
        import sys
        if 'dagster' in sys.modules:
            print("   ⚠️  Dagster déjà chargé")
        
        # Importer code métier
        from src.accidents.bronze.ingest import ingest_caracteristiques
        from src.accidents.silver.features import load_accidents
        from src.accidents.gold.datasets import build_datasets
        
        # Vérifier que Dagster n'a pas été importé
        if 'dagster' not in sys.modules:
            print("   ✅ Code métier indépendant de Dagster")
            return True
        else:
            print("   ⚠️  Dagster importé indirectement")
            return False
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False


def test_schema_constants():
    """Vérifie que les schémas sont correctement définis."""
    print("\n🧪 Test 3/4 : Constantes schémas...")
    
    try:
        from src.accidents.bronze.schema import BRONZE_SCHEMA
        from src.accidents.silver.schema import SILVER_SCHEMA
        from src.accidents.gold.schema import GOLD_SCHEMA
        
        assert BRONZE_SCHEMA == "ducklake.bronze", f"Bronze schema incorrect: {BRONZE_SCHEMA}"
        assert SILVER_SCHEMA == "ducklake.silver", f"Silver schema incorrect: {SILVER_SCHEMA}"
        assert GOLD_SCHEMA == "ducklake.gold", f"Gold schema incorrect: {GOLD_SCHEMA}"
        
        print(f"   ✅ Schémas: {BRONZE_SCHEMA}, {SILVER_SCHEMA}, {GOLD_SCHEMA}")
        return True
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False


def test_pipeline_functions():
    """Vérifie que les fonctions principales sont définies."""
    print("\n🧪 Test 4/4 : Fonctions pipeline...")
    
    try:
        from src.accidents.bronze.ingest import ingest_caracteristiques, ingest_usagers, create_accidents_nc, ingest_all
        from src.accidents.silver.features import load_accidents, load_routes_grid, build_grid, generate_negative_samples, build_feature_store
        from src.accidents.gold.datasets import build_datasets
        
        functions = [
            ingest_caracteristiques,
            ingest_usagers,
            create_accidents_nc,
            ingest_all,
            load_accidents,
            load_routes_grid,
            build_grid,
            generate_negative_samples,
            build_feature_store,
            build_datasets,
        ]
        
        for func in functions:
            assert callable(func), f"{func.__name__} n'est pas callable"
        
        print(f"   ✅ {len(functions)} fonctions pipeline définies")
        return True
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False


def main():
    """Lance tous les tests."""
    print("=" * 60)
    print("🎯 Validation Phase 1 - Structure Code")
    print("=" * 60)
    
    results = [
        test_imports(),
        test_no_dagster_dependency(),
        test_schema_constants(),
        test_pipeline_functions(),
    ]
    
    print("\n" + "=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ SUCCÈS : {passed}/{total} tests passés")
        print("\n📋 Prochaines étapes:")
        print("   1. Configurer .env avec credentials DuckLake")
        print("   2. Tester connexion: python infra/scripts/test-ducklake.py")
        print("   3. Initialiser schémas: python -m src.accidents.setup")
        print("   4. Lancer ingestion: python -m src.accidents.bronze.ingest")
        return 0
    else:
        print(f"❌ ÉCHEC : {passed}/{total} tests passés")
        return 1


if __name__ == "__main__":
    sys.exit(main())
