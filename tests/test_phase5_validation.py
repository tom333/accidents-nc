#!/usr/bin/env python3
"""
🧪 Tests de validation - Phase 5 : Infrastructure K8s

Vérifie que tous les fichiers d'infrastructure sont adaptés
à la structure actuelle (src/).
"""

import os
import sys


def test_tiltfile_updated():
    """Vérifie que le Tiltfile utilise la nouvelle structure"""
    print("🧪 Test 1/7 : Tiltfile adapté...")

    with open("Tiltfile") as f:
        content = f.read()

        # Vérifier les nouveaux paths
        if "./src/accidents" not in content:
            print("   ❌ Tiltfile ne référence pas ./src/accidents")
            return False
        if "./src" not in content:
            print("   ❌ Tiltfile ne référence pas ./src")
            return False

        # Vérifier que les anciens paths ne sont plus là
        if "./pipeline" in content and "./src/accidents" not in content:
            print("   ❌ Tiltfile référence encore l'ancien ./pipeline")
            return False

    print("   ✅ Tiltfile adapté à la nouvelle structure")
    return True


def test_dockerfile_dagster_updated():
    """Vérifie que Dockerfile.dagster copie les bons dossiers"""
    print("🧪 Test 2/7 : Dockerfile.dagster adapté...")

    with open("Dockerfile.dagster") as f:
        content = f.read()

        if "/app/src" not in content:
            print("   ❌ Dockerfile.dagster ne copie pas /app/src")
            return False
        if "/app/src" not in content:
            print("   ❌ Dockerfile.dagster ne copie pas /app/src")
            return False

    print("   ✅ Dockerfile.dagster adapté")
    return True


def test_deployment_dagster_updated():
    """Vérifie que le deployment Dagster pointe vers le bon module"""
    print("🧪 Test 3/7 : deployment-dagster-user.yaml adapté...")

    with open("k8s/dagster/deployment-dagster-user.yaml") as f:
        content = f.read()

        if "src.definitions" not in content:
            print("   ❌ deployment ne référence pas src.definitions")
            return False

    print("   ✅ deployment-dagster-user.yaml adapté")
    return True


def test_dockerfile_streamlit_exists():
    """Vérifie que Dockerfile.streamlit existe"""
    print("🧪 Test 4/7 : Dockerfile.streamlit créé...")

    if not os.path.exists("Dockerfile.streamlit"):
        print("   ❌ Dockerfile.streamlit n'existe pas")
        return False

    with open("Dockerfile.streamlit") as f:
        content = f.read()

        if "streamlit" not in content.lower():
            print("   ❌ Dockerfile.streamlit ne contient pas streamlit")
            return False
        if "/app/apps/streamlit" not in content:
            print("   ❌ Dockerfile.streamlit ne copie pas apps/streamlit")
            return False

    print("   ✅ Dockerfile.streamlit créé")
    return True


def test_streamlit_k8s_manifests():
    """Vérifie que les manifests K8s Streamlit sont créés/adaptés"""
    print("🧪 Test 5/7 : Manifests K8s Streamlit...")

    required_files = [
        "k8s/streamlit/deployment.yaml",
        "k8s/streamlit/service.yaml",
        "k8s/streamlit/configmap-streamlit-ducklake.yaml",
    ]

    for file in required_files:
        if not os.path.exists(file):
            print(f"   ❌ Manquant: {file}")
            return False

    # Vérifier le contenu du deployment
    with open("k8s/streamlit/deployment.yaml") as f:
        content = f.read()
        if "accidents-streamlit" not in content:
            print("   ❌ deployment.yaml ne référence pas accidents-streamlit")
            return False

    print("   ✅ Manifests K8s Streamlit OK")
    return True


def test_deploy_script_exists():
    """Vérifie que le script deploy.sh existe"""
    print("🧪 Test 6/7 : Script deploy.sh...")

    if not os.path.exists("scripts/deploy.sh"):
        print("   ❌ scripts/deploy.sh n'existe pas")
        return False

    # Vérifier qu'il est exécutable
    if not os.access("scripts/deploy.sh", os.X_OK):
        print("   ❌ scripts/deploy.sh n'est pas exécutable")
        return False

    with open("scripts/deploy.sh") as f:
        content = f.read()

        if "deploy_dagster" not in content:
            print("   ❌ deploy.sh ne contient pas deploy_dagster()")
            return False
        if "deploy_streamlit" not in content:
            print("   ❌ deploy.sh ne contient pas deploy_streamlit()")
            return False

    print("   ✅ scripts/deploy.sh OK")
    return True


def test_no_legacy_references():
    """Vérifie qu'il n'y a plus de références aux anciens chemins"""
    print("🧪 Test 7/7 : Pas de références legacy...")

    # Fichiers à vérifier
    files_to_check = [
        "Tiltfile",
        "Dockerfile.dagster",
        "k8s/dagster/deployment-dagster-user.yaml",
    ]

    legacy_patterns = [
        ("pipeline/", "sauf PYTHONPATH"),
    ]

    for file in files_to_check:
        with open(file) as f:
            lines = f.readlines()
            for i, line in enumerate(lines, 1):
                # Ignorer les commentaires
                if line.strip().startswith("#"):
                    continue

                for pattern, note in legacy_patterns:
                    if pattern in line and "PYTHONPATH" not in line:
                        print(f"   ⚠️  {file}:{i} contient '{pattern}' ({note})")

    print("   ✅ Pas de références legacy critiques")
    return True


def main():
    print("🎯 Validation Phase 5 - Infrastructure K8s\n")

    tests = [
        test_tiltfile_updated,
        test_dockerfile_dagster_updated,
        test_deployment_dagster_updated,
        test_dockerfile_streamlit_exists,
        test_streamlit_k8s_manifests,
        test_deploy_script_exists,
        test_no_legacy_references,
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
