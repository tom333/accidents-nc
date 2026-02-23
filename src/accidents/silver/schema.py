"""Schémas silver (données enrichies et nettoyées)."""

# Nom du schéma silver dans DuckLake
SILVER_SCHEMA = "ducklake.silver"

# Tables silver:
# - silver.full_dataset : Accidents + négatifs + features enrichies
#   Colonnes: datetime, latitude, longitude, atm, hour, dayofweek, month, target,
#            is_weekend, is_rush_morning, is_rush_evening, is_night,
#            hour_sin, hour_cos, dayofweek_sin, dayofweek_cos,
#            road_type, speed_limit, is_holiday, school_holidays
