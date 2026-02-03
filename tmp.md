env:
  - name: DUCKLAKE_DATABASE_URL
    value: "postgres:dbname=ducklake_catalog host=postgresql.datalab.svc.cluster.local user=dagster password=..."
  - name: DUCKLAKE_DATA_PATH
    value: "s3://accidents-bucket/ducklake/"
  - name: AWS_ENDPOINT_URL
    value: "http://rustfs-svc.ia-lab.svc.cluster.local:9000"
  - name: AWS_ACCESS_KEY_ID
    valueFrom:
      secretKeyRef:
        name: rustfs-credentials
        key: access-key
  - name: AWS_SECRET_ACCESS_KEY
    valueFrom:
      secretKeyRef:
        name: rustfs-credentials
        key: secret-key
  - name: AWS_REGION
    value: "us-east-1"