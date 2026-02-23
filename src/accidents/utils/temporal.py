"""Utilitaires temporels."""
from __future__ import annotations

import numpy as np
import pandas as pd


# Jours fériés Nouvelle-Calédonie (mois, jour)
NC_HOLIDAYS = {
    (1, 1),   # Nouvel An
    (5, 1),   # Fête du Travail
    (5, 8),   # Victoire 1945
    (7, 14),  # Fête Nationale
    (9, 24),  # Fête de la Citoyenneté
    (11, 1),  # Toussaint
    (11, 11), # Armistice 1918
    (12, 25), # Noël
}

# Mois de vacances scolaires NC
SCHOOL_HOLIDAY_MONTHS = {1, 7, 8, 12}


def cyclical_encode(values: pd.Series | np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Encodage cyclique sin/cos pour features temporelles.
    
    Args:
        values: Valeurs à encoder (ex: heures 0-23, jour semaine 0-6)
        period: Période du cycle (24 pour heures, 7 pour jours)
        
    Returns:
        Tuple (sin_values, cos_values)
    """
    values_array = np.array(values)
    sin_values = np.sin(2 * np.pi * values_array / period)
    cos_values = np.cos(2 * np.pi * values_array / period)
    return sin_values, cos_values


def add_cyclical_features(df: pd.DataFrame) -> None:
    """
    Ajoute les features cycliques au DataFrame (in-place).
    
    Args:
        df: DataFrame avec colonnes 'hour' et 'dayofweek'
    """
    df['hour_sin'], df['hour_cos'] = cyclical_encode(df['hour'], 24)
    df['dayofweek_sin'], df['dayofweek_cos'] = cyclical_encode(df['dayofweek'], 7)


def is_nc_holiday(month: int, day: int) -> bool:
    """Vérifie si une date est un jour férié en Nouvelle-Calédonie."""
    return (month, day) in NC_HOLIDAYS


def is_school_holiday(month: int) -> bool:
    """Vérifie si le mois correspond aux vacances scolaires NC."""
    return month in SCHOOL_HOLIDAY_MONTHS


def add_nc_holidays(df: pd.DataFrame) -> None:
    """
    Ajoute les features de jours fériés et vacances scolaires (in-place).
    
    Args:
        df: DataFrame avec colonnes 'month' et 'datetime'
    """
    df['is_holiday'] = df.apply(
        lambda row: int(is_nc_holiday(row['month'], row['datetime'].day)), 
        axis=1
    )
    df['school_holidays'] = df['month'].apply(lambda m: int(is_school_holiday(m)))


def is_rush_hour(hour: int, period: str = 'morning') -> bool:
    """
    Vérifie si l'heure correspond à une heure de pointe.
    
    Args:
        hour: Heure (0-23)
        period: 'morning' (7-9h) ou 'evening' (17-19h)
        
    Returns:
        True si heure de pointe
    """
    if period == 'morning':
        return 7 <= hour <= 9
    elif period == 'evening':
        return 17 <= hour <= 19
    else:
        raise ValueError("period doit être 'morning' ou 'evening'")


def add_rush_hour_features(df: pd.DataFrame) -> None:
    """
    Ajoute les features d'heures de pointe (in-place).
    
    Args:
        df: DataFrame avec colonne 'hour'
    """
    df['is_rush_morning'] = df['hour'].apply(lambda h: int(is_rush_hour(h, 'morning')))
    df['is_rush_evening'] = df['hour'].apply(lambda h: int(is_rush_hour(h, 'evening')))
    df['is_night'] = df['hour'].apply(lambda h: int(h >= 22 or h <= 6))


def add_weekend_feature(df: pd.DataFrame) -> None:
    """
    Ajoute la feature weekend (in-place).
    
    Args:
        df: DataFrame avec colonne 'dayofweek'
    """
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)


def compute_temporal_features(df: pd.DataFrame) -> None:
    """
    Calcule toutes les features temporelles (in-place).
    
    Args:
        df: DataFrame avec colonnes 'hour', 'dayofweek', 'month', 'datetime'
    """
    add_weekend_feature(df)
    add_rush_hour_features(df)
    add_cyclical_features(df)
    add_nc_holidays(df)
