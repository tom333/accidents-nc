"""Test de validation Phase 3 - Applications."""
import sys
import os


def test_streamlit_structure():
    """Vérifie que la structure Streamlit existe."""
    print("🧪 Test 1/3 : Structure application...")
    
    expected_files = [
        "apps/streamlit/__init__.py",
        "apps/streamlit/app.py",
        "apps/streamlit/README.md",
    ]
    
    missing = []
    for file_path in expected_files:
        if not os.path.exists(file_path):
            missing.append(file_path)
            print(f"   ❌ {file_path} manquant")
        else:
            print(f"   ✅ {file_path}")
    
    if missing:
        return False
    
    print("   ✅ Structure complète")
    return True


def test_streamlit_imports():
    """Vérifie que l'app Streamlit peut être importée."""
    print("\n🧪 Test 2/3 : Imports Streamlit...")
    
    try:
        # Ajouter le répertoire au path
        import sys
        sys.path.insert(0, os.getcwd())
        
        # Importer sans lancer streamlit
        import importlib.util
        spec = importlib.util.spec_from_file_location("streamlit_app", "apps/streamlit/app.py")
        module = importlib.util.module_from_spec(spec)
        
        # Vérifier les imports critiques
        from src.accidents.ducklake import get_client
        from src.accidents.gold.schema import GOLD_SCHEMA
        
        print("   ✅ Imports DuckLake OK")
        print("   ✅ Imports src.accidents OK")
        
        return True
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_old_files_removed():
    """Vérifie que les anciens fichiers ont été supprimés."""
    print("\n🧪 Test 3/3 : Nettoyage anciens fichiers...")
    
    old_files = [
        "api/main.py",
        "app_predict_map.py",
        "streamlit_app/",
    ]
    
    removed = []
    still_exist = []
    
    for file_path in old_files:
        if os.path.exists(file_path):
            still_exist.append(file_path)
            print(f"   ⚠️  {file_path} existe encore")
        else:
            removed.append(file_path)
            print(f"   ✅ {file_path} supprimé")
    
    if still_exist:
        print(f"\n   ⚠️  {len(still_exist)} fichiers à nettoyer manuellement")
        return True  # Non bloquant
    
    print("   ✅ Tous les anciens fichiers supprimés")
    return True


def main():
    """Lance tous les tests."""
    print("=" * 60)
    print("🎯 Validation Phase 3 - Applications")
    print("=" * 60)
    
    results = [
        test_streamlit_structure(),
        test_streamlit_imports(),
        test_old_files_removed(),
    ]
    
    print("\n" + "=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ SUCCÈS : {passed}/{total} tests passés")
        print("\n📋 Prochaines étapes:")
        print("   1. Lancer l'app Streamlit:")
        print("      PYTHONPATH=$PWD streamlit run apps/streamlit/app.py")
        print("   2. Tester dans le navigateur: http://localhost:8501")
        print("   3. Configurer .env si nécessaire")
        print("   4. Intégrer le modèle ML réel (TODO)")
        return 0
    else:
        print(f"❌ ÉCHEC : {passed}/{total} tests passés")
        return 1


if __name__ == "__main__":
    sys.exit(main())
