import json

import numpy as np
import optuna
import torch
import torch.nn as nn
from dagster import AssetExecutionContext, asset
from sklearn.metrics import auc as auc_metric
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.assets.gold.datasets import FEATURE_COLUMNS, N_GEO_CLUSTERS
from src.assets.gold.schema import GOLD_SCHEMA
from src.ducklake import get_client
from src.utils.models import save_model, save_predictions


class MLPEmbeddingClassifier(nn.Module):
    """
    MLP avec Entity Embeddings pour geo_cluster.

    Architecture:
    - Embedding layer pour geo_cluster (catégoriel)
    - Dense layers pour features numériques
    - Concatenation + MLP final
    """

    def __init__(self, num_features_cont, num_clusters, embed_dim, hidden1, hidden2, dropout_rate):
        super().__init__()

        # Embedding pour transformer cluster ID (0-49) en vecteur dense
        self.embedding = nn.Embedding(num_clusters, embed_dim)

        # Réseau Dense combiné (Features continues + Embedding)
        input_dim_total = num_features_cont + embed_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim_total, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )

        self.scaler = None  # Stocké pour predict_proba

    def forward(self, x_cat, x_cont):
        """
        Forward pass.

        Args:
            x_cat: Tensor de shape (batch, 1) - geo_cluster IDs
            x_cont: Tensor de shape (batch, num_features_cont) - features continues
        """
        x_embedded = self.embedding(x_cat.squeeze(1))  # (batch, embed_dim)
        x_combined = torch.cat([x_embedded, x_cont], dim=1)
        return self.network(x_combined).squeeze()

    def predict_proba(self, X):
        """Compatibilité sklearn pour blending."""
        self.eval()
        with torch.no_grad():
            # Séparer geo_cluster (index 0) et features continues
            geo_cluster = torch.LongTensor(X[:, 0].astype(int)).unsqueeze(1)

            # Scaler les features continues
            X_cont = X[:, 1:]
            if self.scaler is not None:
                X_cont = self.scaler.transform(X_cont)
            X_cont_tensor = torch.FloatTensor(X_cont)

            probs = self.forward(geo_cluster, X_cont_tensor).numpy()
            return np.column_stack([1 - probs, probs])


@asset(group_name="gold", deps=["ml_datasets"], required_resource_keys={"mlflow"})
def tune_mlp(context: AssetExecutionContext):
    """
    Entraîne un MLP avec Entity Embeddings pour geo_cluster.

    Features:
    - geo_cluster (catégoriel) → Embedding layer
    - Autres features → Normalisation StandardScaler
    """
    conn = get_client().conn
    train_df = conn.execute(f"SELECT * FROM {GOLD_SCHEMA}.train").df()
    test_df = conn.execute(f"SELECT * FROM {GOLD_SCHEMA}.test").df()

    X_train_full = train_df[FEATURE_COLUMNS].values
    y_train = train_df["target"].values
    X_test_full = test_df[FEATURE_COLUMNS].values
    y_test = test_df["target"].values

    # Séparer geo_cluster (col 0) et features continues (col 1+)
    geo_train = X_train_full[:, 0].astype(int)
    X_train_cont = X_train_full[:, 1:]

    geo_test = X_test_full[:, 0].astype(int)
    X_test_cont = X_test_full[:, 1:]

    # Normaliser features continues uniquement
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_cont)
    X_test_scaled = scaler.transform(X_test_cont)

    context.log.info(
        f"🧠 MLP Embeddings: geo_cluster={N_GEO_CLUSTERS} zones, features_cont={X_train_scaled.shape[1]}"
    )

    def objective(trial):
        # Hyperparamètres
        embed_dim = trial.suggest_int("embed_dim", 4, 16)
        hidden1 = trial.suggest_int("hidden1", 64, 256)
        hidden2 = trial.suggest_int("hidden2", 32, 128)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

        model = MLPEmbeddingClassifier(
            num_features_cont=X_train_scaled.shape[1],
            num_clusters=N_GEO_CLUSTERS,
            embed_dim=embed_dim,
            hidden1=hidden1,
            hidden2=hidden2,
            dropout_rate=dropout,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        # Training rapide (15 epochs pour Optuna)
        for _epoch in range(15):
            model.train()
            for i in range(0, len(X_train_scaled), batch_size):
                geo_batch = torch.LongTensor(geo_train[i : i + batch_size]).unsqueeze(1)
                X_batch = torch.FloatTensor(X_train_scaled[i : i + batch_size])
                y_batch = torch.FloatTensor(y_train[i : i + batch_size])

                optimizer.zero_grad()
                output = model(geo_batch, X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

        # Évaluation
        model.eval()
        with torch.no_grad():
            geo_test_tensor = torch.LongTensor(geo_test).unsqueeze(1)
            X_test_tensor = torch.FloatTensor(X_test_scaled)
            probs = model(geo_test_tensor, X_test_tensor).numpy()
            preds = (probs > 0.5).astype(int)

        return recall_score(y_test, preds)

    study = optuna.create_study(
        direction="maximize",
        study_name="MLP_Embeddings",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    context.log.info(f"✅ Best trial: {study.best_trial.value:.4f}")
    context.log.info(f"📊 Best params: {study.best_params}")

    # Entraîner modèle final avec meilleurs hyperparamètres
    best_params = study.best_params
    final_model = MLPEmbeddingClassifier(
        num_features_cont=X_train_scaled.shape[1],
        num_clusters=N_GEO_CLUSTERS,
        embed_dim=best_params["embed_dim"],
        hidden1=best_params["hidden1"],
        hidden2=best_params["hidden2"],
        dropout_rate=best_params["dropout"],
    )
    final_model.scaler = scaler  # Stocker pour predict_proba

    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params["lr"])
    criterion = nn.BCELoss()
    batch_size = best_params["batch_size"]

    # Training complet (30 epochs)
    context.log.info("🔄 Training final model...")
    for _epoch in range(30):
        final_model.train()
        for i in range(0, len(X_train_scaled), batch_size):
            geo_batch = torch.LongTensor(geo_train[i : i + batch_size]).unsqueeze(1)
            X_batch = torch.FloatTensor(X_train_scaled[i : i + batch_size])
            y_batch = torch.FloatTensor(y_train[i : i + batch_size])

            optimizer.zero_grad()
            output = final_model(geo_batch, X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    # Prédictions finales
    preds = final_model.predict_proba(X_test_full)

    # MLflow logging via resource Dagster
    mlflow = context.resources.mlflow
    mlflow.log_params(best_params)

    y_pred = (preds[:, 1] > 0.5).astype(int)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, preds[:, 1])
    auc_val = auc_metric([0, 1], [recall, precision])

    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("auc", auc_val)
    mlflow.log_metric("auc_roc", auc_roc)

    save_model(final_model, "mlp_model.pkl")
    mlflow.log_artifact("mlp_model.pkl")

    save_predictions(preds, "mlp_preds.csv")
    mlflow.log_artifact("mlp_preds.csv")

    mlp_metrics = {
        "model_path": "mlp_model.pkl",
        "preds_path": "mlp_preds.csv",
        "auc_roc": auc_roc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }
    with open("mlp_metrics.json", "w") as f:
        json.dump(mlp_metrics, f)
    mlflow.log_artifact("mlp_metrics.json")

    return mlp_metrics
