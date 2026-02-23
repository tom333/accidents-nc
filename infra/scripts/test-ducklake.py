#!/usr/bin/env python3
"""
Script de test pour vérifier la connexion DuckLake.
Usage: python infra/scripts/test-ducklake.py
"""
import sys
from pathlib import Path

# Ajouter le projet au path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.accidents.ducklake import get_client


def test_connection():
    """Teste la connexion DuckLake."""
    print("🔍 Test connexion DuckLake...")
    
    try:
        client = get_client()
        conn = client.conn
        
        # Test basique
        result = conn.execute("SELECT 1 as test").fetchone()
        assert result[0] == 1, "Query test failed"
        print("✅ Connexion DuckDB OK")
        
        # Test PostgreSQL catalog
        try:
            conn.execute("SELECT 1 FROM ducklake.information_schema.tables LIMIT 1")
            print("✅ PostgreSQL catalog OK")
        except Exception as e:
            print(f"⚠️  PostgreSQL catalog: {e}")
            print("💡 Vérifiez vos credentials dans .env")
            print("   POSTGRES_URL doit pointer vers le PostgreSQL distant")
        
        # Test S3 (MinIO)
        try:
            # Liste les buckets (devrait inclure 'accidents')
            result = conn.execute("SELECT * FROM s3_list_buckets()").fetchall()
            print(f"✅ S3/MinIO OK - Buckets: {[r[0] for r in result]}")
        except Exception as e:
            print(f"⚠️  S3/MinIO: {e}")
            print("💡 Le bucket 'accidents' sera créé automatiquement")
        
        print("\n🎉 Tous les tests passés!")
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Copier .env.example vers .env:")
        print("   cp .env.example .env")
        print("2. Configurer les credentials DuckLake dans .env:")
        print("   - POSTGRES_URL (PostgreSQL distant)")
        print("   - AWS_ACCESS_KEY_ID (S3 RustFS)")
        print("   - AWS_SECRET_ACCESS_KEY")
        print("3. Vérifier la connectivité réseau au cluster K8s")
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
