from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import joblib
import numpy as np
import osmnx as ox
import pandas as pd
import polars as pl
from scipy.spatial.distance import cdist
from shapely.geometry import Point

from .config import SILVER_SCHEMA, PIPELINE_PARAMS, BRONZE_SCHEMA, ensure_connection

ROUTES_CACHE = Path("routes_with_features.pkl")
ROUTES_GEOJSON = Path("routes.nc")


def _load_accidents() -> pl.DataFrame:
    conn = ensure_connection()
    df = conn.execute(f"SELECT * FROM {BRONZE_SCHEMA}.accidents_nc").pl()
    print(f"📊 Accidents bruts chargés: {len(df)} lignes")
    filtered = df.filter(
        (pl.col('latitude').is_not_null()) &
        (pl.col('longitude').is_not_null()) &
        (pl.col('latitude').is_between(-23.0, -19.5)) &
        (pl.col('longitude').is_between(163.5, 168.0))
    )
    print(f"✅ Après filtrage géographique: {len(filtered)} lignes")
    return filtered


def _load_routes_grid(areas: Tuple[str, ...]) -> gpd.GeoDataFrame:
    if ROUTES_CACHE.exists():
        print("📦 Chargement routes_with_features.pkl")
        routes_data = joblib.load(ROUTES_CACHE)
        if isinstance(routes_data, pd.DataFrame):
            routes_df = routes_data
        elif isinstance(routes_data, dict):
            if 'routes_grid' in routes_data:
                routes_df = routes_data['routes_grid']
            else:
                routes_df = pd.DataFrame(routes_data)
        else:
            routes_df = pd.DataFrame(routes_data)
        return gpd.GeoDataFrame(
            routes_df,
            geometry=gpd.points_from_xy(routes_df['longitude'], routes_df['latitude']),
            crs="EPSG:4326"
        )

    print("⚠️  routes_with_features.pkl absent → génération depuis OSM")
    buffer_file = ROUTES_GEOJSON
    if buffer_file.exists():
        return gpd.read_file(buffer_file)

    all_edges = []
    for place in areas:
        try:
            print(f"⬇️  Téléchargement OSM: {place}")
            G = ox.graph_from_place(place, network_type='drive')
            edges = ox.graph_to_gdfs(G, nodes=False)
            all_edges.append(edges)
        except Exception as exc:
            print(f"   ⚠️  Échec {place}: {exc}")

    if not all_edges:
        raise RuntimeError("Impossible de générer les routes.")

    routes = gpd.GeoDataFrame(pd.concat(all_edges, ignore_index=True), crs=all_edges[0].crs)
    routes = routes.drop_duplicates(subset="geometry")
    routes.to_file(buffer_file, driver="GeoJSON")
    print(f"✅ Routes enregistrées dans {buffer_file}")
    return routes


AREAS = (
    "Boulouparis, New Caledonia",
    "Bourail, New Caledonia",
    "Canala, New Caledonia",
    "Dumbéa, New Caledonia",
    "Farino, New Caledonia",
    "Hienghène, New Caledonia",
    "Houaïlou, New Caledonia",
    "Île des Pins, New Caledonia",
    "Kaala-Gomen, New Caledonia",
    "Koné, New Caledonia",
    "Kouaoua, New Caledonia",
    "Koumac, New Caledonia",
    "La Foa, New Caledonia",
    "Lifou, New Caledonia",
    "Maré, New Caledonia",
    "Moindou, New Caledonia",
    "Mont-Dore, New Caledonia",
    "Nouméa, New Caledonia",
    "Ouégoa, New Caledonia",
    "Ouvéa, New Caledonia",
    "Païta, New Caledonia",
    "Poindimié, New Caledonia",
    "Ponérihouen, New Caledonia",
    "Poya, New Caledonia",
    "Sarraméa, New Caledonia",
    "Thio, New Caledonia",
    "Touho, New Caledonia",
    "Voh, New Caledonia",
    "Yaté, New Caledonia",
    "Belep, New Caledonia",
    "Îles Loyauté, New Caledonia",
    "Province Nord, New Caledonia",
    "Province Sud, New Caledonia",
)


def _build_grid(routes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if {'road_type', 'speed_limit', 'accident_density_5km', 'dist_to_noumea_km'} <= set(routes.columns):
        return routes

    step = PIPELINE_PARAMS.grid_step
    buffer_meters = PIPELINE_PARAMS.buffer_meters
    lat_min, lat_max = -23.0, -19.5
    lon_min, lon_max = 163.5, 168.0
    lats = np.arange(lat_min, lat_max, step)
    lons = np.arange(lon_min, lon_max, step)
    grid = pd.DataFrame([(lat, lon) for lat in lats for lon in lons], columns=['latitude', 'longitude'])
    grid['geometry'] = grid.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    grid_gdf = gpd.GeoDataFrame(grid, geometry='geometry', crs="EPSG:4326").to_crs(epsg=3857)

    routes_buffer = routes.to_crs(epsg=3857).buffer(buffer_meters)
    routes_buffer_gdf = gpd.GeoDataFrame(geometry=routes_buffer, crs="EPSG:3857")
    grid_on_roads = gpd.sjoin(grid_gdf, routes_buffer_gdf, how="inner", predicate="intersects").drop(columns="index_right")
    return grid_on_roads.to_crs(epsg=4326)


def _generate_negative_samples(accidents: pl.DataFrame, routes_grid: gpd.GeoDataFrame) -> pl.DataFrame:
    params = PIPELINE_PARAMS
    n_samples = len(accidents) * params.n_negative_samples_ratio
    if len(routes_grid) == 0:
        sampled = pl.DataFrame({
            'datetime': pl.Series(np.random.choice(accidents['datetime'], size=n_samples)).to_pandas(),
            'latitude': np.random.choice(accidents['latitude'], size=n_samples),
            'longitude': np.random.choice(accidents['longitude'], size=n_samples),
            'atm': np.random.choice(accidents['atm'], size=n_samples),
        })
    else:
        accidents_pd = accidents.to_pandas()
        accidents_gdf = gpd.GeoDataFrame(
            accidents_pd,
            geometry=gpd.points_from_xy(accidents_pd['longitude'], accidents_pd['latitude']),
            crs="EPSG:4326"
        )
        buffer_m = params.accident_exclusion_buffer_km * 1000
        accidents_buffer = accidents_gdf.to_crs(epsg=3857).buffer(buffer_m)
        accidents_buffer_gdf = gpd.GeoDataFrame(geometry=accidents_buffer, crs="EPSG:3857")

        routes_grid_fixed = gpd.GeoDataFrame(
            routes_grid.drop(columns='geometry'),
            geometry=routes_grid.geometry.values,
            crs=routes_grid.crs
        ).to_crs(epsg=3857)

        safe_routes = routes_grid_fixed.sjoin(accidents_buffer_gdf, how="left", predicate="intersects")
        safe_routes = safe_routes[safe_routes['index_right'].isna()].drop(columns=['index_right'])
        safe_routes = safe_routes.to_crs(epsg=4326)

        if len(safe_routes) == 0:
            safe_routes = routes_grid

        route_sample_indices = np.random.choice(
            len(safe_routes),
            size=min(n_samples, len(safe_routes) * 10),
            replace=True
        )
        route_sample = safe_routes.iloc[route_sample_indices]

        n_from_accidents = int(len(route_sample) * params.temporal_risk_ratio)
        n_random = len(route_sample) - n_from_accidents
        date_range = accidents['datetime'].max() - accidents['datetime'].min()
        random_timestamps = [
            accidents['datetime'].min() + np.timedelta64(
                int(np.random.uniform(0, date_range.total_seconds())), 's'
            )
            for _ in range(n_random)
        ]
        combined_timestamps = np.concatenate([
            np.random.choice(accidents['datetime'], size=n_from_accidents),
            random_timestamps
        ])
        np.random.shuffle(combined_timestamps)

        sampled = pl.DataFrame({
            'datetime': pd.to_datetime(combined_timestamps),
            'latitude': route_sample['latitude'].values,
            'longitude': route_sample['longitude'].values,
            'atm': np.random.choice(accidents['atm'], size=len(route_sample)),
        })

    sampled = sampled.with_columns([
        pl.col('datetime').dt.hour().alias('hour'),
        pl.col('datetime').dt.weekday().alias('dayofweek'),
        pl.col('datetime').dt.month().alias('month'),
        pl.lit(0).alias('target')
    ])
    return sampled


def _compute_interactions(df: pd.DataFrame) -> None:
    df['lat_hour'] = df['latitude'] * df['hour'] / 24
    df['lon_hour'] = df['longitude'] * df['hour'] / 24
    df['lat_dayofweek'] = df['latitude'] * df['dayofweek'] / 7
    df['lon_dayofweek'] = df['longitude'] * df['dayofweek'] / 7
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_rush_morning'] = ((df['hour'] >= 6) & (df['hour'] <= 8)).astype(int)
    df['is_rush_evening'] = ((df['hour'] >= 16) & (df['hour'] <= 18)).astype(int)
    df['is_night'] = ((df['hour'] >= 19) | (df['hour'] <= 5)).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)


def _attach_precomputed_features(df: pd.DataFrame, routes_grid: gpd.GeoDataFrame) -> None:
    required_cols = ['road_type', 'speed_limit', 'accident_density_5km', 'dist_to_noumea_km']
    if not all(col in routes_grid.columns for col in required_cols):
        for col, default in zip(required_cols, [2, 50, 0, 100]):
            df[col] = default
        return

    full_coords = df[['latitude', 'longitude']].values
    grid_coords = routes_grid[['latitude', 'longitude']].values
    arrays = {col: (routes_grid[col].to_numpy() if hasattr(routes_grid[col], 'to_numpy') else routes_grid[col].values)
              for col in required_cols}
    batch_size = 1000
    selections: Dict[str, list] = {col: [] for col in required_cols}
    for start in range(0, len(full_coords), batch_size):
        batch = full_coords[start:start + batch_size]
        distances = cdist(batch, grid_coords)
        nearest = distances.argmin(axis=1)
        for col in required_cols:
            selections[col].extend(arrays[col][nearest])
    for col in required_cols:
        df[col] = selections[col]


def _add_temporal_features(df: pd.DataFrame) -> None:
    nc_holidays = {(1, 1), (5, 1), (5, 8), (7, 14), (9, 24), (11, 1), (11, 11), (12, 25)}
    df['is_holiday'] = df.apply(lambda row: int((row['month'], row['datetime'].day) in nc_holidays), axis=1)
    df['school_holidays'] = df['month'].isin([1, 7, 8, 12]).astype(int)


def build_feature_store() -> dict[str, int]:
    accidents = _load_accidents()
    routes = _load_routes_grid(AREAS)
    routes_grid = _build_grid(routes)
    negative_samples = _generate_negative_samples(accidents, routes_grid)

    conn = ensure_connection()
    conn.register('accidents_tbl', accidents.to_pandas())
    conn.register('negatives_tbl', negative_samples.to_pandas())
    combined = conn.execute(
        """
        SELECT * FROM accidents_tbl
        UNION ALL
        SELECT * FROM negatives_tbl
        """
    ).df()
    conn.unregister('accidents_tbl')
    conn.unregister('negatives_tbl')

    _compute_interactions(combined)
    _attach_precomputed_features(combined, routes_grid)
    _add_temporal_features(combined)

    conn.register('features_dataframe', combined)
    conn.execute(f"CREATE OR REPLACE TABLE {SILVER_SCHEMA}.full_dataset AS SELECT * FROM features_dataframe")
    conn.unregister('features_dataframe')

    print(f"💾 Table {SILVER_SCHEMA}.full_dataset écrite ({len(combined)} lignes)")
    return {
        'rows': len(combined),
        'negatives': len(negative_samples),
        'positives': len(accidents),
    }
