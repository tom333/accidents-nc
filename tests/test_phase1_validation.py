"""Test de validation Phase 1 - Structure et imports."""

import sys


def test_imports():
    """Vérifie que tous les imports fonctionnent."""
    print("🧪 Test 1/4 : Imports...")

    try:
        from src.config import DuckLakeConfig  # noqa: F401
        from src.resources.ducklake import get_client  # noqa: F401

        print("   ✅ Core (ducklake, config)")

        from src.assets.bronze.ingest import ingest_all  # noqa: F401
        from src.assets.bronze.schema import BRONZE_SCHEMA  # noqa: F401

        print("   ✅ Bronze (ingest, schema)")

        from src.assets.silver.features import build_feature_store  # noqa: F401
        from src.assets.silver.schema import SILVER_SCHEMA  # noqa: F401

        print("   ✅ Silver (features, schema)")

        from src.assets.gold.datasets import build_datasets  # noqa: F401
        from src.assets.gold.schema import GOLD_SCHEMA  # noqa: F401

        print("   ✅ Gold (datasets, schema)")

        from src.assets.utils import (
            buffer_routes,  # noqa: F401
            compute_classification_metrics,  # noqa: F401
            create_spatial_grid,  # noqa: F401
            cyclical_encode,  # noqa: F401
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

        if "dagster" in sys.modules:
            print("   ⚠️  Dagster déjà chargé")

        # Importer code métier

        # Vérifier que Dagster n'a pas été importé
        if "dagster" not in sys.modules:
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
        from src.assets.bronze.schema import BRONZE_SCHEMA
        from src.assets.gold.schema import GOLD_SCHEMA
        from src.assets.silver.schema import SILVER_SCHEMA

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
        from src.assets.bronze.ingest import (
            create_accidents_nc,
            ingest_all,
            ingest_caracteristiques,
            ingest_usagers,
        )
        from src.assets.gold.datasets import build_datasets
        from src.assets.silver.features import (
            build_feature_store,
            build_grid,
            generate_negative_samples,
            load_accidents,
            load_routes_grid,
        )

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
        print("   3. Initialiser schémas: python -m src.setup")
        print("   4. Lancer ingestion: python -m src.accidents.bronze.ingest")
        return 0
    else:
        print(f"❌ ÉCHEC : {passed}/{total} tests passés")
        return 1


if __name__ == "__main__":
    sys.exit(main())
