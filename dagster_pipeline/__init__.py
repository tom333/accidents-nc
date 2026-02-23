"""Dagster package for accidents pipeline."""

from dagster import Definitions
from .definitions import defs

__all__ = ["defs"]
