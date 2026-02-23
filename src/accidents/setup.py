"""
Script d'initialisation des schémas DuckLake.

Crée les schémas Bronze, Silver, Gold dans le catalogue PostgreSQL.
À exécuter une seule fois lors de la première mise en place.
"""

from src.accidents.ducklake import get_client
from src.accidents.bronze.schema import BRONZE_SCHEMA
from src.accidents.silver.schema import SILVER_SCHEMA
from src.accidents.gold.schema import GOLD_SCHEMA


def init_schemas():
    """Initialise les schémas Bronze, Silver, Gold dans DuckLake."""
    client = get_client()
    conn = client.conn
    
    schemas = [BRONZE_SCHEMA, SILVER_SCHEMA, GOLD_SCHEMA]
    
    for schema in schemas:
        print(f"📦 Création du schéma {schema}...")
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        print(f"   ✅ {schema} créé/vérifié")
    
    print("\n✨ Tous les schémas sont initialisés !")
    
    # Vérification
    print("\n📊 Schémas disponibles :")
    result = conn.execute(
        "SELECT schema_name FROM ducklake.information_schema.schemata "
        "WHERE schema_name IN ('bronze', 'silver', 'gold')"
    ).fetchall()
    for row in result:
        print(f"   - {row[0]}")


if __name__ == "__main__":
    init_schemas()
