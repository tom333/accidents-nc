#!/usr/bin/env python3
"""
Créer le bucket MinIO pour FastTrackML
"""
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

print("📦 Création du bucket FastTrackML dans MinIO\n")

# Configuration
endpoint = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'https://rustfs.tgu.ovh')
access_key = os.getenv('AWS_ACCESS_KEY_ID')
secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
bucket_name = 'mlflow-artifacts'

print(f"Endpoint: {endpoint}")
print(f"Access Key: {access_key}")
print(f"Bucket: {bucket_name}\n")

# Créer le client S3
s3 = boto3.client(
    's3',
    endpoint_url=endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key
)

try:
    # Vérifier si le bucket existe
    s3.head_bucket(Bucket=bucket_name)
    print(f"✅ Le bucket '{bucket_name}' existe déjà")
except:
    # Créer le bucket
    try:
        s3.create_bucket(Bucket=bucket_name)
        print(f"✅ Bucket '{bucket_name}' créé avec succès")
    except Exception as e:
        print(f"❌ Erreur lors de la création du bucket: {e}")
        exit(1)

# Lister tous les buckets pour confirmation
print("\n📋 Buckets disponibles:")
buckets = s3.list_buckets()
for bucket in buckets['Buckets']:
    print(f"   - {bucket['Name']}")

print("\n✅ Configuration MinIO prête pour FastTrackML !")
