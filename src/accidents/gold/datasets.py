"""Préparation des datasets ML (gold layer)."""
from __future__ import annotations

import os
from pathlib import Path

import boto3
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..ducklake import get_client
from ..silver.schema import SILVER_SCHEMA
from .schema import GOLD_SCHEMA

# Configuration S3
S3_BUCKET = os.getenv("S3_BUCKET", "accidents-bucket")
S3_CACHE_PREFIX = "cache/"
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "https://rustfs.tgu.ovh")

# Features utilisées pour ML
FEATURE_COLUMNS = [
    'latitude', 'longitude',
    'hour', 'dayofweek', 'month',
    'atm',
    'is_weekend', 'is_rush_morning', 'is_rush_evening', 'is_night',
    'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
    # 'road_type', 'speed_limit',
    'is_holiday', 'school_holidays'
]

# Paramètres split
TEST_SIZE = 0.2
RANDOM_STATE = 42


def _get_s3_client():
    """Créer un client S3 configuré."""
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )


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


def build_datasets() -> dict[str, int]:
    """Crée gold.train et gold.test depuis silver.full_dataset."""
    print("🎯 Construction datasets ML...")
    
    client = get_client()
    conn = client.conn
    
    # Charger données silver
    df = conn.execute(f"SELECT * FROM {SILVER_SCHEMA}.full_dataset").df()
    print(f"📊 Données silver: {len(df)} lignes")
    
    # Encoder atm
    le = LabelEncoder()
    df['atm'] = le.fit_transform(df['atm'].astype(str))
    
    # Filtrer colonnes et supprimer NA
    dataset = df.dropna(subset=FEATURE_COLUMNS)
    print(f"✅ Après drop NA: {len(dataset)} lignes")
    
    X = dataset[FEATURE_COLUMNS]
    y = dataset['target']
    
    # Split train/test stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    # Créer DataFrames finaux
    train_df = X_train.copy()
    train_df['target'] = y_train.values
    
    test_df = X_test.copy()
    test_df['target'] = y_test.values
    
    # Sauvegarder dans DuckLake
    conn.register('train_tmp', train_df)
    conn.register('test_tmp', test_df)
    
    conn.execute(f"CREATE OR REPLACE TABLE {GOLD_SCHEMA}.train AS SELECT * FROM train_tmp")
    conn.execute(f"CREATE OR REPLACE TABLE {GOLD_SCHEMA}.test AS SELECT * FROM test_tmp")
    
    conn.unregister('train_tmp')
    conn.unregister('test_tmp')
    
    # Sauvegarder metadata
    feature_meta = pd.DataFrame({
        'feature_name': FEATURE_COLUMNS,
        'order_index': range(len(FEATURE_COLUMNS))
    })
    
    conn.register('meta_tmp', feature_meta)
    conn.execute(f"CREATE OR REPLACE TABLE {GOLD_SCHEMA}.feature_metadata AS SELECT * FROM meta_tmp")
    conn.unregister('meta_tmp')
    
    # Sauvegarder encoders localement et sur S3
    joblib.dump(le, 'atm_encoder.pkl')
    joblib.dump(FEATURE_COLUMNS, 'features.pkl')
    
    _upload_to_s3(Path('atm_encoder.pkl'), f"{S3_CACHE_PREFIX}atm_encoder.pkl")
    _upload_to_s3(Path('features.pkl'), f"{S3_CACHE_PREFIX}features.pkl")
    
    print(f"💾 Tables gold.train ({len(train_df)}) et gold.test ({len(test_df)}) créées")
    print(f"💾 Encoders sauvegardés: atm_encoder.pkl, features.pkl")
    
    return {
        'train_rows': len(train_df),
        'test_rows': len(test_df),
        'total_features': len(FEATURE_COLUMNS),
        'train_positives': int(y_train.sum()),
        'test_positives': int(y_test.sum()),
    }
