"""Utilitaires géospatiaux."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


def buffer_routes(gdf: gpd.GeoDataFrame, buffer_meters: float = 200) -> gpd.GeoDataFrame:
    """
    Applique un buffer aux routes.

    Args:
        gdf: GeoDataFrame des routes en EPSG:4326
        buffer_meters: Buffer en mètres

    Returns:
        GeoDataFrame avec buffer appliqué
    """
    # Reprojeter en Web Mercator (mètres)
    gdf_projected = gdf.to_crs(epsg=3857)

    # Appliquer buffer
    gdf_buffered = gdf_projected.copy()
    gdf_buffered.geometry = gdf_projected.geometry.buffer(buffer_meters)

    # Retour WGS84
    return gdf_buffered.to_crs(epsg=4326)


def create_spatial_grid(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    step: float = 0.02,
    clip_to: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """
    Crée une grille spatiale régulière.

    Args:
        minx, miny, maxx, maxy: Bounding box en degrés
        step: Résolution de la grille en degrés
        clip_to: GeoDataFrame optionnel pour cliper la grille

    Returns:
        GeoDataFrame avec les points de la grille
    """
    lats = np.arange(miny, maxy, step)
    lons = np.arange(minx, maxx, step)

    grid_points = [Point(lon, lat) for lat in lats for lon in lons]
    grid_gdf = gpd.GeoDataFrame(geometry=grid_points, crs="EPSG:4326")

    if clip_to is not None:
        # Filtrer dans le buffer
        grid_gdf = grid_gdf[grid_gdf.within(clip_to.unary_union)]

    # Extraire coordonnées
    grid_gdf["latitude"] = grid_gdf.geometry.y
    grid_gdf["longitude"] = grid_gdf.geometry.x

    return grid_gdf.reset_index(drop=True)


def spatial_exclusion_filter(
    candidate_coords: np.ndarray, exclusion_coords: np.ndarray, exclusion_distance_km: float = 0.3
) -> np.ndarray:
    """
    Filtre les points trop proches des coordonnées d'exclusion.

    Args:
        candidate_coords: Array (N, 2) de coordonnées candidates [lat, lon]
        exclusion_coords: Array (M, 2) de coordonnées à éviter [lat, lon]
        exclusion_distance_km: Distance minimale en km

    Returns:
        Masque booléen (N,) indiquant les points valides
    """
    from scipy.spatial.distance import cdist

    # Distance euclidienne en degrés * 111 km/degré (approximation)
    distances_km = cdist(candidate_coords, exclusion_coords, metric="euclidean") * 111

    # Points valides = éloignés de tous les points d'exclusion
    safe_mask = (distances_km > exclusion_distance_km).all(axis=1)

    return safe_mask


def extract_osm_features(routes_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Extrait les features OSM pertinentes.

    Args:
        routes_gdf: GeoDataFrame OSM avec attributs

    Returns:
        DataFrame avec road_type et speed_limit
    """
    features = pd.DataFrame()

    if "highway" in routes_gdf.columns:
        features["road_type"] = routes_gdf["highway"].fillna("unknown")
    else:
        features["road_type"] = "unknown"

    if "maxspeed" in routes_gdf.columns:
        features["speed_limit"] = (
            pd.to_numeric(routes_gdf["maxspeed"].str.extract(r"(\d+)")[0], errors="coerce")
            .fillna(50)
            .astype(int)
        )
    else:
        features["speed_limit"] = 50

    return features
