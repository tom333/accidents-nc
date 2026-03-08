"""Test de validation Phase 2 - Assets Dagster."""

import sys


def test_dagster_loads():
    """Vérifie que Dagster charge correctement."""
    print("🧪 Test 1/3 : Chargement Dagster...")

    try:
        print("   ✅ Definitions chargées")
        return True
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False


def test_assets_defined():
    """Vérifie que les assets sont définis."""
    print("\n🧪 Test 2/3 : Assets définis...")

    try:
        from src.definitions import all_assets

        asset_keys = [asset.key.path[-1] for asset in all_assets]
        expected = ["caracteristiques", "usagers", "accidents_nc", "full_dataset", "ml_datasets"]

        for asset_name in expected:
            if asset_name in asset_keys:
                print(f"   ✅ {asset_name}")
            else:
                print(f"   ❌ {asset_name} manquant")
                return False

        print(f"\n   ✅ {len(asset_keys)} assets définis")
        return True
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_asset_dependencies():
    """Vérifie le graphe de dépendances."""
    print("\n🧪 Test 3/3 : Dépendances assets...")

    try:
        from src.definitions import all_assets

        # Vérifier les assets avec deps explicites
        deps_found = {
            "accidents_nc": False,
            "full_dataset": False,
            "ml_datasets": False,
        }

        for asset in all_assets:
            asset_name = asset.key.path[-1]

            if asset_name == "accidents_nc":
                # Doit dépendre de caracteristiques et usagers
                if len(asset.asset_deps) >= 2:
                    deps_found["accidents_nc"] = True
                    print("   ✅ accidents_nc a des dépendances")

            elif asset_name == "full_dataset":
                # Doit dépendre de accidents_nc
                if len(asset.asset_deps) >= 1:
                    deps_found["full_dataset"] = True
                    print("   ✅ full_dataset a des dépendances")

            elif asset_name == "ml_datasets":
                # Doit dépendre de full_dataset
                if len(asset.asset_deps) >= 1:
                    deps_found["ml_datasets"] = True
                    print("   ✅ ml_datasets a des dépendances")

        if all(deps_found.values()):
            print("\n   ✅ Graphe de dépendances présent")
            return True
        else:
            print(f"\n   ❌ Dépendances manquantes: {[k for k, v in deps_found.items() if not v]}")
            return False
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Lance tous les tests."""
    print("=" * 60)
    print("🎯 Validation Phase 2 - Assets Dagster")
    print("=" * 60)

    results = [
        test_dagster_loads(),
        test_assets_defined(),
        test_asset_dependencies(),
    ]

    print("\n" + "=" * 60)

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✅ SUCCÈS : {passed}/{total} tests passés")
        print("\n📋 Prochaines étapes:")
        print("   1. Démarrer Dagster dev:")
        print("      PYTHONPATH=$PWD dagster dev -f src/definitions.py")
        print("   2. Ouvrir http://localhost:3000")
        print("   3. Visualiser le graphe des assets")
        print("   4. Matérialiser les assets (si .env configuré)")
        return 0
    else:
        print(f"❌ ÉCHEC : {passed}/{total} tests passés")
        return 1


if __name__ == "__main__":
    sys.exit(main())
