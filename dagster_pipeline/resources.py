"""Dagster resources."""

from contextlib import contextmanager
from dagster import ConfigurableResource
from src.accidents.ducklake import get_client


class DuckLakeResource(ConfigurableResource):
    """Resource pour accéder au client DuckLake."""
    
    def get_connection(self):
        """Retourne le client DuckLake."""
        return get_client()
