from __future__ import annotations

import os
from pathlib import Path
import boto3
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import (
    GOLD_SCHEMA,
    SILVER_SCHEMA,
    PIPELINE_PARAMS,
    ensure_connection,
)

# Configuration S3
S3_BUCKET = "accidents-bucket"
S3_CACHE_PREFIX = "cache/"
S3_ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "https://rustfs.tgu.ovh")


def _get_s3_client():
    """Créer un client S3 configuré pour RustFS"""
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )


def _upload_to_s3(local_path: Path, s3_key: str) -> bool:
    """Uploader un fichier local vers S3"""
    try:
        s3_client = _get_s3_client()
        s3_client.upload_file(str(local_path), S3_BUCKET, s3_key)
        print(f"📤 Uploadé vers S3: {local_path} → s3://{S3_BUCKET}/{s3_key}")
        return True
    except Exception as e:
        print(f"⚠️  Erreur upload S3: {e}")
        return False

FEATURE_COLUMNS = [
    'latitude', 'longitude',
    'hour', 'dayofweek', 'month',
    'atm',
    # 'lat_hour', 'lon_hour', 'lat_dayofweek', 'lon_dayofweek',
    'is_weekend', 'is_rush_morning', 'is_rush_evening', 'is_night',
    'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
    'road_type', 'speed_limit',
    # 'accident_density_5km', 'dist_to_noumea_km',
    'is_holiday', 'school_holidays'
]


def build_datasets() -> dict[str, int]:
    conn = ensure_connection()
    df = conn.execute(f"SELECT * FROM {SILVER_SCHEMA}.full_dataset").df()
    le = LabelEncoder()
    df['atm'] = le.fit_transform(df['atm'].astype(str))

    dataset = df.dropna(subset=FEATURE_COLUMNS)
    X = dataset[FEATURE_COLUMNS]
    y = dataset['target']

    params = PIPELINE_PARAMS
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params.test_size,
        random_state=params.random_state,
        stratify=y
    )

    train_df = X_train.copy()
    train_df['target'] = y_train.values
    test_df = X_test.copy()
    test_df['target'] = y_test.values

    conn.register('train_tmp', train_df)
    conn.register('test_tmp', test_df)
    conn.execute(f"CREATE OR REPLACE TABLE {GOLD_SCHEMA}.train AS SELECT * FROM train_tmp")
    conn.execute(f"CREATE OR REPLACE TABLE {GOLD_SCHEMA}.test AS SELECT * FROM test_tmp")
    conn.unregister('train_tmp')
    conn.unregister('test_tmp')

    feature_meta = pd.DataFrame({'feature_name': FEATURE_COLUMNS, 'order_index': range(len(FEATURE_COLUMNS))})
    conn.register('meta_tmp', feature_meta)
    conn.execute(f"CREATE OR REPLACE TABLE {GOLD_SCHEMA}.feature_metadata AS SELECT * FROM meta_tmp")
    conn.unregister('meta_tmp')

    joblib.dump(le, 'atm_encoder.pkl')
    joblib.dump(FEATURE_COLUMNS, 'features.pkl')
    
    # Uploader vers S3
    _upload_to_s3(Path('atm_encoder.pkl'), f"{S3_CACHE_PREFIX}atm_encoder.pkl")
    _upload_to_s3(Path('features.pkl'), f"{S3_CACHE_PREFIX}features.pkl")

    return {
        'train_rows': len(train_df),
        'test_rows': len(test_df),
        'n_features': len(FEATURE_COLUMNS),
    }
