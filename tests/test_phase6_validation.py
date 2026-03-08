#!/usr/bin/env python3
"""
🧪 Tests de validation - Phase 6 : Tests & CI/CD

Vérifie que tous les fichiers CI/CD sont créés et que les tests
unitaires sont en place.
"""

import os
import sys


def test_github_workflows_exist():
    """Vérifie que les workflows GitHub Actions sont créés"""
    print("🧪 Test 1/6 : Workflows GitHub Actions...")

    required_workflows = [
        ".github/workflows/ci-tests.yml",
        ".github/workflows/ci-build.yml",
        ".github/workflows/cd-update-manifests.yml",
    ]

    for workflow in required_workflows:
        if not os.path.exists(workflow):
            print(f"   ❌ Manquant: {workflow}")
            return False
        print(f"   ✅ {workflow}")

    print("   ✅ Tous les workflows créés")
    return True


def test_unit_tests_exist():
    """Vérifie que les tests unitaires sont créés"""
    print("🧪 Test 2/6 : Tests unitaires...")

    required_tests = [
        "tests/unit/test_bronze.py",
        "tests/unit/test_silver.py",
        "tests/unit/test_gold.py",
    ]

    for test in required_tests:
        if not os.path.exists(test):
            print(f"   ❌ Manquant: {test}")
            return False
        print(f"   ✅ {test}")

    print("   ✅ Tous les tests unitaires créés")
    return True


def test_pytest_config():
    """Vérifie la configuration pytest dans pyproject.toml"""
    print("🧪 Test 3/6 : Configuration pytest...")

    if not os.path.exists("pyproject.toml"):
        print("   ❌ pyproject.toml manquant")
        return False

    with open("pyproject.toml") as f:
        content = f.read()

        required_configs = [
            "[tool.pytest.ini_options]",
            "pytest-cov",
            "--cov=src/accidents",
        ]

        for config in required_configs:
            if config not in content:
                print(f"   ❌ Configuration manquante: {config}")
                return False

    print("   ✅ Configuration pytest OK")
    return True


def test_ruff_config():
    """Vérifie la configuration ruff (linter)"""
    print("🧪 Test 4/6 : Configuration ruff...")

    with open("pyproject.toml") as f:
        content = f.read()

        if "[tool.ruff]" not in content:
            print("   ❌ Configuration ruff manquante")
            return False

    print("   ✅ Configuration ruff OK")
    return True


def test_ci_tests_workflow_structure():
    """Vérifie la structure du workflow CI tests"""
    print("🧪 Test 5/6 : Structure workflow CI tests...")

    with open(".github/workflows/ci-tests.yml") as f:
        content = f.read()

        required_jobs = [
            "lint",
            "unit-tests",
            "validation-tests",
        ]

        for job in required_jobs:
            if job not in content:
                print(f"   ❌ Job manquant: {job}")
                return False

        # Vérifier les actions clés
        if "astral-sh/setup-uv@v5" not in content:
            print("   ❌ Action setup-uv manquante")
            return False

        if "pytest" not in content:
            print("   ❌ pytest non utilisé")
            return False

    print("   ✅ Workflow CI tests bien structuré")
    return True


def test_cd_workflow_argocd():
    """Vérifie que le workflow CD mentionne ArgoCD"""
    print("🧪 Test 6/6 : Workflow CD avec ArgoCD...")

    with open(".github/workflows/cd-update-manifests.yml") as f:
        content = f.read()

        if "argocd" not in content.lower():
            print("   ⚠️  ArgoCD non mentionné explicitement")

        # Vérifier update des manifests
        if "k8s/dagster/deployment-dagster-user.yaml" not in content:
            print("   ❌ Update deployment Dagster manquant")
            return False

        if "k8s/streamlit/deployment.yaml" not in content:
            print("   ❌ Update deployment Streamlit manquant")
            return False

        if "git commit" not in content:
            print("   ❌ Commit git manquant")
            return False

    print("   ✅ Workflow CD ArgoCD OK")
    return True


def main():
    print("🎯 Validation Phase 6 - Tests & CI/CD\n")

    tests = [
        test_github_workflows_exist,
        test_unit_tests_exist,
        test_pytest_config,
        test_ruff_config,
        test_ci_tests_workflow_structure,
        test_cd_workflow_argocd,
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
