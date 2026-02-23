#!/usr/bin/env python3
"""
Test de Validation - Phase 7 : Documentation

Vérifie que tous les documents de la Phase 7 sont créés et complets.
"""

import os
import sys

def test_readme_updated():
    """Vérifie que README.md a été mis à jour."""
    print("\n🧪 Test 1/5 : README.md mis à jour...")
    
    readme_path = "README.md"
    assert os.path.exists(readme_path), f"❌ {readme_path} n'existe pas"
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Vérifier sections clés
    required_sections = [
        "# 🚦 Accidents NC",
        "## 📊 Vue d'Ensemble",
        "## 🏗️ Architecture",
        "## 📂 Structure du Projet",
        "## 🚀 Quickstart",
        "## 📖 Documentation",
        "## 🔄 Workflow CI/CD",
        "docs/architecture.md",
        "docs/workflow.md",
        "docs/deployment.md"
    ]
    
    for section in required_sections:
        assert section in content, f"❌ Section '{section}' manquante dans README.md"
    
    print(f"   ✅ README.md complet ({len(content)} caractères)")
    return True

def test_architecture_doc():
    """Vérifie que docs/architecture.md existe et est complet."""
    print("\n🧪 Test 2/5 : docs/architecture.md...")
    
    doc_path = "docs/architecture.md"
    assert os.path.exists(doc_path), f"❌ {doc_path} n'existe pas"
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Vérifier sections clés
    required_sections = [
        "# 🏛️ Architecture du Projet",
        "## 🎯 Vue d'ensemble",
        "## 🥉🥈🥇 Architecture Médaillons",
        "## 🦆 DuckLake",
        "## 🎭 Pipeline Dagster",
        "## ☸️ Infrastructure Kubernetes",
        "## 🌐 Applications",
        "## 🚀 CI/CD GitOps",
        "## 📊 Flux de Données"  # Majuscule à Données
    ]
    
    for section in required_sections:
        assert section in content, f"❌ Section '{section}' manquante dans architecture.md"
    
    # Vérifier contenu substantiel (>20k caractères)
    assert len(content) > 20000, f"❌ architecture.md trop court ({len(content)} caractères)"
    
    print(f"   ✅ architecture.md complet ({len(content)} caractères)")
    return True

def test_workflow_doc():
    """Vérifie que docs/workflow.md existe et est complet."""
    print("\n🧪 Test 3/5 : docs/workflow.md...")
    
    doc_path = "docs/workflow.md"
    assert os.path.exists(doc_path), f"❌ {doc_path} n'existe pas"
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Vérifier sections clés
    required_sections = [
        "# 🔄 Workflow de Développement",
        "## 🚀 Setup Initial",
        "## 💻 Développement Local",
        "## ⚡ Développement avec Tilt",
        "## 🎭 Travailler avec Dagster",
        "## 🔬 Exploration avec Marimo",
        "## 🧪 Tests",
        "## 🌳 Workflow Git",
        "## 🚀 CI/CD",
        "## 🛠️ Troubleshooting"
    ]
    
    for section in required_sections:
        assert section in content, f"❌ Section '{section}' manquante dans workflow.md"
    
    # Vérifier contenu substantiel (>15k caractères)
    assert len(content) > 15000, f"❌ workflow.md trop court ({len(content)} caractères)"
    
    print(f"   ✅ workflow.md complet ({len(content)} caractères)")
    return True

def test_deployment_doc():
    """Vérifie que docs/deployment.md existe et est complet."""
    print("\n🧪 Test 4/5 : docs/deployment.md...")
    
    doc_path = "docs/deployment.md"
    assert os.path.exists(doc_path), f"❌ {doc_path} n'existe pas"
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Vérifier sections clés
    required_sections = [
        "# 🚢 Guide de Déploiement",
        "## ✅ Prérequis",
        "## 🏗️ Architecture de Déploiement",
        "## ☸️ Setup Kubernetes",
        "## 🦆 Configuration DuckLake",
        "## 🎭 Déploiement Dagster",
        "## 📊 Déploiement Streamlit",
        "## 🔄 Configuration ArgoCD",
        "## 🚀 CI/CD Pipeline",
        "## 📊 Monitoring et Logs",
        "## 🛠️ Troubleshooting Production"
    ]
    
    for section in required_sections:
        assert section in content, f"❌ Section '{section}' manquante dans deployment.md"
    
    # Vérifier contenu substantiel (>20k caractères)
    assert len(content) > 20000, f"❌ deployment.md trop court ({len(content)} caractères)"
    
    print(f"   ✅ deployment.md complet ({len(content)} caractères)")
    return True

def test_docs_links_in_readme():
    """Vérifie que README.md contient les bons liens vers les docs."""
    print("\n🧪 Test 5/5 : Liens docs dans README...")
    
    readme_path = "README.md"
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    required_links = [
        "[docs/architecture.md](docs/architecture.md)",
        "[docs/workflow.md](docs/workflow.md)",
        "[docs/deployment.md](docs/deployment.md)",
        "[TILT.md](TILT.md)",
        "[DOCKER.md](DOCKER.md)",
        "[INDUSTRIALISATION.md](INDUSTRIALISATION.md)"
    ]
    
    for link in required_links:
        assert link in content, f"❌ Lien '{link}' manquant dans README.md"
    
    print(f"   ✅ Tous les liens docs présents dans README.md")
    return True

def main():
    """Execute tous les tests de validation Phase 7."""
    print("="*50)
    print("🎯 Validation Phase 7 - Documentation")
    print("="*50)
    
    tests = [
        test_readme_updated,
        test_architecture_doc,
        test_workflow_doc,
        test_deployment_doc,
        test_docs_links_in_readme,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ ÉCHEC : {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ ERREUR : {e}")
            failed += 1
    
    print("\n" + "="*50)
    if failed == 0:
        print(f"✅ SUCCÈS : {passed}/{len(tests)} tests passés")
        print("="*50)
        return 0
    else:
        print(f"❌ ÉCHEC : {passed}/{len(tests)} tests passés, {failed} échecs")
        print("="*50)
        return 1

if __name__ == "__main__":
    sys.exit(main())
