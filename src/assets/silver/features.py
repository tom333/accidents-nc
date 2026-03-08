"""Feature engineering et enrichissement (silver layer)."""

from __future__ import annotations

import os
from pathlib import Path

import boto3
import geopandas as gpd
import joblib
import numpy as np
import osmnx as ox
import pandas as pd
import polars as pl
from botocore.exceptions import ClientError
from scipy.spatial import cKDTree

from src.ducklake import get_client
from src.utils.spatial import buffer_routes, create_spatial_grid, spatial_exclusion_filter
from src.utils.temporal import compute_temporal_features

from ..bronze.schema import BRONZE_SCHEMA
from .schema import SILVER_SCHEMA

# Fichiers cache
ROUTES_CACHE = Path("routes_with_features.pkl")
ROUTES_GEOJSON = Path("routes.nc")

# Configuration S3
S3_BUCKET = os.getenv("S3_BUCKET", "accidents-bucket")
S3_CACHE_PREFIX = "cache/"
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "https://rustfs.tgu.ovh")

# Communes de Nouvelle-Calédonie
AREAS = (
    "Boulouparis, New Caledonia",
    "Bourail, New Caledonia",
    "Canala, New Caledonia",
    "Dumbéa, New Caledonia",
    "Farino, New Caledonia",
    "Hienghène, New Caledonia",
    "Houaïlou, New Caledonia",
    "Kaala-Gomen, New Caledonia",
    "Koné, New Caledonia",
    "Koumac, New Caledonia",
    "La Foa, New Caledonia",
    "Lifou, New Caledonia",
    "Maré, New Caledonia",
    "Moindou, New Caledonia",
    "Mont-Dore, New Caledonia",
    "Nouméa, New Caledonia",
    "Ouvéa, New Caledonia",
    "Païta, New Caledonia",
    "Poindimié, New Caledonia",
    "Ponérihouen, New Caledonia",
    "Pouébo, New Caledonia",
    "Pouembout, New Caledonia",
    "Poya, New Caledonia",
    "Sarraméa, New Caledonia",
    "Thio, New Caledonia",
    "Touho, New Caledonia",
    "Voh, New Caledonia",
    "Yaté, New Caledonia",
)

# Paramètres (à déplacer dans config.py si besoin)
BUFFER_METERS = 200
GRID_STEP = 0.02
ACCIDENT_EXCLUSION_BUFFER_KM = 0.3
TEMPORAL_RISK_RATIO = 0.85
MAX_NEGATIVE_SAMPLES_MULTIPLIER = 100


def _get_s3_client():
    """Créer un client S3 configuré."""
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )


def _download_from_s3(s3_key: str, local_path: Path) -> bool:
    """Télécharger un fichier depuis S3."""
    try:
        s3_client = _get_s3_client()
        s3_client.download_file(S3_BUCKET, s3_key, str(local_path))
        print(f"📥 Téléchargé depuis S3: s3://{S3_BUCKET}/{s3_key}")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print(f"⚠️  Fichier absent sur S3: s3://{S3_BUCKET}/{s3_key}")
        return False
    except Exception as e:
        print(f"⚠️  Erreur téléchargement S3: {e}")
        return False


def _upload_to_s3(local_path: Path, s3_key: str) -> bool:
    """Uploader un fichier local vers S3."""
    try:
        s3_client = _get_s3_client()
        s3_client.upload_file(str(local_path), S3_BUCKET, s3_key)
        print(f"📤 Uploadé vers S3: s3://{S3_BUCKET}/{s3_key}")
        return True
    except Exception as e:
        print(f"⚠️  Erreur upload S3: {e}")
        return False


def load_accidents() -> pl.DataFrame:
    """Charge les accidents depuis bronze.accidents_nc avec filtrage géographique."""
    client = get_client()
    conn = client.conn

    df = conn.execute(f"SELECT * FROM {BRONZE_SCHEMA}.accidents_nc").pl()
    print(f"📊 Accidents bruts: {len(df)} lignes")

    filtered = df.filter(
        (pl.col("latitude").is_not_null())
        & (pl.col("longitude").is_not_null())
        & (pl.col("latitude").is_between(-23.0, -19.5))
        & (pl.col("longitude").is_between(163.5, 168.0))
    )
    print(f"✅ Après filtrage géographique: {len(filtered)} lignes")
    return filtered


def load_routes_grid(areas: tuple[str, ...] = AREAS) -> gpd.GeoDataFrame:
    """Charge ou génère la grille routière depuis OSM."""
    # Essayer cache .pkl d'abord
    s3_cache_key = f"{S3_CACHE_PREFIX}{ROUTES_CACHE.name}"
    if not ROUTES_CACHE.exists():
        _download_from_s3(s3_cache_key, ROUTES_CACHE)

    if ROUTES_CACHE.exists():
        print(f"📦 Chargement {ROUTES_CACHE}")
        routes_data = joblib.load(ROUTES_CACHE)
        if isinstance(routes_data, pd.DataFrame):
            routes_df = routes_data
        elif isinstance(routes_data, dict):
            routes_df = routes_data.get("routes_grid", pd.DataFrame(routes_data))
        else:
            routes_df = pd.DataFrame(routes_data)

        return gpd.GeoDataFrame(
            routes_df,
            geometry=gpd.points_from_xy(routes_df["longitude"], routes_df["latitude"]),
            crs="EPSG:4326",
        )

    # Sinon essayer GeoJSON
    print(f"⚠️  {ROUTES_CACHE} absent → vérification {ROUTES_GEOJSON}")
    s3_geojson_key = f"{S3_CACHE_PREFIX}{ROUTES_GEOJSON.name}"
    if not ROUTES_GEOJSON.exists():
        _download_from_s3(s3_geojson_key, ROUTES_GEOJSON)

    if ROUTES_GEOJSON.exists():
        routes = gpd.read_file(ROUTES_GEOJSON)
        _upload_to_s3(ROUTES_GEOJSON, s3_geojson_key)
        return routes

    # Sinon télécharger depuis OSM
    print("⚠️  routes.nc absent → génération depuis OSM")
    all_edges = []
    for place in areas:
        try:
            print(f"⬇️  Téléchargement OSM: {place}")
            G = ox.graph_from_place(place, network_type="drive")
            edges = ox.graph_to_gdfs(G, nodes=False)
            all_edges.append(edges)
        except Exception as exc:
            print(f"   ⚠️  Échec {place}: {exc}")

    if not all_edges:
        raise RuntimeError("Impossible de générer les routes OSM")

    routes = gpd.GeoDataFrame(pd.concat(all_edges, ignore_index=True), crs=all_edges[0].crs)
    routes = routes.drop_duplicates(subset="geometry")
    routes.to_file(ROUTES_GEOJSON, driver="GeoJSON")
    print(f"✅ Routes enregistrées: {ROUTES_GEOJSON}")

    _upload_to_s3(ROUTES_GEOJSON, s3_geojson_key)
    return routes


def build_grid(routes: gpd.GeoDataFrame) -> pd.DataFrame:
    """Construit une grille de points sur les routes avec buffer."""
    print("🗺️  Construction grille routière...")

    # Buffer des routes
    buffered_gdf = buffer_routes(routes, BUFFER_METERS)

    # Bounding box
    minx, miny, maxx, maxy = buffered_gdf.total_bounds

    # Créer grille
    grid_gdf = create_spatial_grid(minx, miny, maxx, maxy, GRID_STEP, clip_to=buffered_gdf)

    # Features OSM par défaut
    grid_gdf["road_type"] = "unknown"
    grid_gdf["speed_limit"] = 50

    result = grid_gdf[["latitude", "longitude", "road_type", "speed_limit"]].reset_index(drop=True)
    print(f"✅ {len(result)} points sur grille")

    return pd.DataFrame(result)


def generate_negative_samples(accidents: pl.DataFrame, routes_grid: pd.DataFrame) -> pl.DataFrame:
    """Génère des échantillons négatifs spatialement sûrs."""
    print("🎲 Génération échantillons négatifs...")

    print(f"🔍 DEBUG: generate_negative_samples accidents columns: {accidents.columns}")
    # Exclure zone buffer autour accidents
    accident_coords = accidents.select(["latitude", "longitude"]).to_numpy()
    grid_coords = routes_grid[["latitude", "longitude"]].to_numpy()

    safe_mask = spatial_exclusion_filter(grid_coords, accident_coords, ACCIDENT_EXCLUSION_BUFFER_KM)
    safe_grid = routes_grid[safe_mask].reset_index(drop=True)

    print(f"   Grille sûre: {len(safe_grid)} points (exclusion {ACCIDENT_EXCLUSION_BUFFER_KM}km)")

    # Nombre d'échantillons négatifs
    n_samples = min(len(safe_grid) * MAX_NEGATIVE_SAMPLES_MULTIPLIER, len(accidents) * 2)

    # distribution temporelle
    n_risk_hours = int(n_samples * TEMPORAL_RISK_RATIO)
    n_random_hours = n_samples - n_risk_hours

    # Échantillonner timestamps depuis accidents (heures à risque)
    accident_datetimes = accidents["datetime"].to_pandas().tolist()
    risk_timestamps = np.random.choice(accident_datetimes, size=n_risk_hours, replace=True)

    # Timestamps aléatoires
    min_date = accidents["datetime"].min()
    max_date = accidents["datetime"].max()
    min_ts = int(min_date.timestamp() * 1e9)
    max_ts = int(max_date.timestamp() * 1e9)
    if min_ts >= max_ts:
        random_timestamps = pd.Series([min_date] * n_random_hours)
    else:
        random_timestamps = pd.to_datetime(
            np.random.randint(min_ts, max_ts, size=n_random_hours), unit="ns"
        )

    all_timestamps = np.concatenate([risk_timestamps, random_timestamps])
    dt_index = pd.to_datetime(all_timestamps)
    # Ensure homogenous types (pydatetime or Timestamp)
    all_timestamps_clean = dt_index.to_pydatetime()

    # Échantillonner points sur grille sûre
    sampled_grid = safe_grid.sample(n=n_samples, replace=True, random_state=42).reset_index(
        drop=True
    )

    # Créer DataFrame
    negatives = pl.DataFrame(
        {
            "datetime": all_timestamps_clean,
            "latitude": sampled_grid["latitude"].to_numpy(),
            "longitude": sampled_grid["longitude"].to_numpy(),
            "atm": np.random.choice(accidents["atm"].to_pandas().dropna().tolist(), size=n_samples),
            "hour": dt_index.hour.values,
            "dayofweek": dt_index.dayofweek.values,
            "month": dt_index.month.values,
            "target": np.zeros(n_samples, dtype=int),
        }
    )

    print(
        f"✅ Négatifs générés: {len(negatives)} ({n_risk_hours} risque + {n_random_hours} aléatoire)"
    )
    return negatives


def _attach_road_features(combined: pd.DataFrame, routes_grid: pd.DataFrame) -> pd.DataFrame:
    """Associe les features routieres via le point de grille le plus proche."""
    if routes_grid.empty:
        raise ValueError("routes_grid est vide, impossible d'enrichir les routes")

    route_coords = routes_grid[["latitude", "longitude"]].to_numpy()
    sample_coords = combined[["latitude", "longitude"]].to_numpy()

    tree = cKDTree(route_coords)
    _, idx = tree.query(sample_coords, k=1)
    nearest = routes_grid.iloc[idx].reset_index(drop=True)

    combined = combined.copy()
    combined["road_type"] = nearest["road_type"].to_numpy()
    combined["speed_limit"] = nearest["speed_limit"].to_numpy()
    return combined


def build_feature_store() -> dict[str, int]:
    """Construit silver.full_dataset : accidents + négatifs + features enrichies."""
    print("🏗️  Construction feature store...")

    # Charger données
    accidents = load_accidents()
    routes = load_routes_grid(AREAS)
    routes_grid = build_grid(routes)

    # Générer négatifs
    negative_samples = generate_negative_samples(accidents, routes_grid)

    # Préparer accidents avec colonnes cohérentes
    accidents_prepared = accidents.select(
        [
            pl.col("datetime"),
            pl.col("latitude"),
            pl.col("longitude"),
            pl.col("atm"),
        ]
    ).with_columns(
        [
            pl.col("datetime").dt.hour().alias("hour"),
            pl.col("datetime").dt.weekday().alias("dayofweek"),
            pl.col("datetime").dt.month().alias("month"),
            pl.lit(1).alias("target"),
        ]
    )

    # Combiner
    client = get_client()
    conn = client.conn

    conn.register("accidents_tbl", accidents_prepared.to_pandas())
    conn.register("negatives_tbl", negative_samples.to_pandas())

    combined = conn.execute("""
        SELECT * FROM accidents_tbl
        UNION ALL
        SELECT * FROM negatives_tbl
    """).df()

    # Enrichissement temporel final (pour avoir toutes les features sur tout le dataset)
    compute_temporal_features(combined)

    # Enrichir features routieres via la grille
    combined = _attach_road_features(combined, routes_grid)

    # Sauvegarder dans DuckLake
    conn.register("features_dataframe", combined)
    conn.execute(
        f"CREATE OR REPLACE TABLE {SILVER_SCHEMA}.full_dataset AS SELECT * FROM features_dataframe"
    )
    conn.unregister("features_dataframe")

    print(f"💾 Table {SILVER_SCHEMA}.full_dataset créée ({len(combined)} lignes)")

    return {
        "rows": len(combined),
        "positives": len(accidents),
        "negatives": len(negative_samples),
    }
