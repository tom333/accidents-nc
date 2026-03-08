#!/usr/bin/env python3
"""
🧪 Tests de validation - Phase 4 : Exploration Notebooks

Vérifie que tous les notebooks Marimo d'exploration sont créés,
utilisent DuckLake, et que les anciens fichiers sont archivés.
"""

import os
import sys


def test_exploration_structure():
    """Vérifie la structure du dossier exploration/"""
    print("🧪 Test 1/4 : Structure exploration/...")

    required_files = [
        "exploration/01_eda_accidents.py",
        "exploration/02_features_engineering.py",
        "exploration/03_model_experiments.py",
        "exploration/04_viz_predictions.py",
    ]

    for file in required_files:
        if not os.path.exists(file):
            print(f"   ❌ Manquant: {file}")
            return False
        print(f"   ✅ {file}")

    print("   ✅ Structure complète")
    return True


def test_notebooks_use_ducklake():
    """Vérifie que les notebooks utilisent DuckLake"""
    print("🧪 Test 2/4 : Import DuckLake...")

    notebooks = [
        "exploration/01_eda_accidents.py",
        "exploration/02_features_engineering.py",
        "exploration/03_model_experiments.py",
        "exploration/04_viz_predictions.py",
    ]

    for notebook in notebooks:
        with open(notebook) as f:
            content = f.read()
            if "from src.resources.ducklake import get_client" not in content:
                print(f"   ❌ {notebook} n'importe pas DuckLake")
                return False
            if "get_client()" not in content:
                print(f"   ❌ {notebook} n'utilise pas get_client()")
                return False

    print("   ✅ Tous les notebooks utilisent DuckLake")
    return True


def test_notebooks_are_marimo():
    """Vérifie que ce sont des notebooks Marimo"""
    print("🧪 Test 3/4 : Format Marimo...")

    notebooks = [
        "exploration/01_eda_accidents.py",
        "exploration/02_features_engineering.py",
        "exploration/03_model_experiments.py",
        "exploration/04_viz_predictions.py",
    ]

    for notebook in notebooks:
        with open(notebook) as f:
            content = f.read()
            if "import marimo" not in content:
                print(f"   ❌ {notebook} n'est pas un notebook Marimo")
                return False
            if "@app.cell" not in content:
                print(f"   ❌ {notebook} n'a pas de cellules Marimo")
                return False

    print("   ✅ Tous les notebooks sont au format Marimo")
    return True


def test_old_files_archived():
    """Vérifie que les anciens fichiers sont archivés"""
    print("🧪 Test 4/4 : Archivage anciens fichiers...")

    # Vérifier que notebooks/ n'existe plus
    if os.path.exists("notebooks/"):
        print("   ❌ notebooks/ existe encore (devrait être archivé)")
        return False
    print("   ✅ notebooks/ archivé")

    # Vérifier que accident_fetch_data.py n'existe plus
    if os.path.exists("accident_fetch_data.py"):
        print("   ❌ accident_fetch_data.py existe encore (devrait être archivé)")
        return False
    print("   ✅ accident_fetch_data.py archivé")

    # Vérifier que archive/ existe
    if not os.path.exists("archive/"):
        print("   ❌ archive/ n'existe pas")
        return False
    print("   ✅ archive/ créé")

    return True


def main():
    print("🎯 Validation Phase 4 - Exploration Notebooks\n")

    tests = [
        test_exploration_structure,
        test_notebooks_use_ducklake,
        test_notebooks_are_marimo,
        test_old_files_archived,
    ]

    results = [test() for test in tests]

    print(f"\n{'=' * 50}")
    if all(results):
        print(f"✅ SUCCÈS : {len(results)}/{len(results)} tests passés")
        return 0
    else:
        failed = len([r for r in results if not r])
        print(f"❌ ÉCHEC : {failed}/{len(results)} tests échoués")
        return 1


if __name__ == "__main__":
    sys.exit(main())
