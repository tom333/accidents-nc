import json

import mlflow.pytorch
import numpy as np
import optuna
import torch
import torch.nn as nn
from dagster import AssetExecutionContext, asset
from mlflow.models import infer_signature
from sklearn.metrics import auc as auc_metric
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.assets.gold.datasets import FEATURE_COLUMNS
from src.assets.gold.schema import GOLD_SCHEMA
from src.ducklake import get_client
from src.utils.models import save_model, save_predictions


class MLPEmbeddingClassifier(nn.Module):
    """
    MLP avec Entity Embeddings pour toutes les features catégorielles.

    Architecture:
    - Embedding layers individuels pour chaque feature catégorielle
    - Dense layers (avec GELU) pour features numériques et embeddings concaténés
    """

    def __init__(self, cat_dims, num_features_cont, hidden1, hidden2, dropout_rate):
        super().__init__()

        # Embeddings pour chaque variable catégorielle
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_categories, emb_dim) for num_categories, emb_dim in cat_dims]
        )

        total_emb_dim = sum(emb_dim for _, emb_dim in cat_dims)
        input_dim_total = total_emb_dim + num_features_cont

        self.network = nn.Sequential(
            nn.Linear(input_dim_total, hidden1),
            nn.GELU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )

        self.scaler = None  # Stocké pour predict_proba
        self.cat_indices = None
        self.cont_indices = None

    def forward(self, x_cat, x_cont):
        """
        Forward pass.

        Args:
            x_cat: Tensor de shape (batch, num_cat_features)
            x_cont: Tensor de shape (batch, num_features_cont)
        """
        emb_outputs = []
        for i, emb_layer in enumerate(self.embeddings):
            emb_outputs.append(emb_layer(x_cat[:, i]))

        if emb_outputs:
            x_embedded = torch.cat(emb_outputs, dim=1)
            x_combined = torch.cat([x_embedded, x_cont], dim=1)
        else:
            x_combined = x_cont

        return self.network(x_combined).squeeze(-1)

    def predict_proba(self, X):
        """Compatibilité sklearn pour blending."""
        self.eval()
        with torch.no_grad():
            x_cat = torch.LongTensor(X[:, self.cat_indices].astype(int))
            x_cont = X[:, self.cont_indices]

            if self.scaler is not None:
                x_cont = self.scaler.transform(x_cont)
            x_cont_tensor = torch.FloatTensor(x_cont)

            probs = self.forward(x_cat, x_cont_tensor).numpy()
            return np.column_stack([1 - probs, probs])


@asset(group_name="gold", deps=["ml_datasets"], required_resource_keys={"mlflow"})
def tune_mlp(context: AssetExecutionContext):
    """
    Entraîne un MLP avec Entity Embeddings pour les features catégorielles.
    """
    registered_model_name = "accidents_mlp"

    conn = get_client().conn
    train_df = conn.execute(f"SELECT * FROM {GOLD_SCHEMA}.train").df()
    test_df = conn.execute(f"SELECT * FROM {GOLD_SCHEMA}.test").df()

    X_train_full = train_df[FEATURE_COLUMNS].values
    y_train = train_df["target"].values
    X_test_full = test_df[FEATURE_COLUMNS].values
    y_test = test_df["target"].values

    # Identifier features catégorielles et continues
    cat_cols = ["geo_cluster", "atm", "road_type", "lit", "surface", "oneway"]
    cat_indices = [FEATURE_COLUMNS.index(c) for c in cat_cols if c in FEATURE_COLUMNS]
    cont_indices = [i for i, c in enumerate(FEATURE_COLUMNS) if i not in cat_indices]

    # Définir dimensions d'embeddings (règle min(50, num_categories // 2))
    cat_dims_base = []
    for c_idx in cat_indices:
        max_val = max(int(np.max(X_train_full[:, c_idx])), int(np.max(X_test_full[:, c_idx])))
        num_cat = max_val + 1
        emb_dim = min(50, max(2, num_cat // 2))
        cat_dims_base.append((num_cat, emb_dim))

    context.log.info(
        f"🧠 MLP Embeddings: {len(cat_indices)} variables cat, {len(cont_indices)} variables cont"
    )

    def objective(trial):
        # Hyperparamètres
        hidden1 = trial.suggest_int("hidden1", 64, 256)
        hidden2 = trial.suggest_int("hidden2", 32, 128)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

        # Prévention Data Leakage: Split interne (Train/Validation)
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train_full, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        scaler = StandardScaler()
        X_fit_cont = scaler.fit_transform(X_fit[:, cont_indices])
        X_val_cont = scaler.transform(X_val[:, cont_indices])

        X_fit_cat = X_fit[:, cat_indices].astype(int)
        X_val_cat = X_val[:, cat_indices].astype(int)

        scale_weight_fit = np.sum(y_fit == 0) / np.sum(y_fit == 1)

        model = MLPEmbeddingClassifier(
            cat_dims=cat_dims_base,
            num_features_cont=len(cont_indices),
            hidden1=hidden1,
            hidden2=hidden2,
            dropout_rate=dropout,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCELoss(reduction="none")

        epochs = 15
        for epoch in range(epochs):
            model.train()
            for i in range(0, len(X_fit), batch_size):
                batch_cat = torch.LongTensor(X_fit_cat[i : i + batch_size])
                batch_cont = torch.FloatTensor(X_fit_cont[i : i + batch_size])
                batch_y = torch.FloatTensor(y_fit[i : i + batch_size])
                weights = torch.FloatTensor(
                    np.where(y_fit[i : i + batch_size] == 1, scale_weight_fit, 1.0)
                )

                optimizer.zero_grad()
                output = model(batch_cat, batch_cont)
                loss = (criterion(output, batch_y) * weights).mean()
                loss.backward()
                optimizer.step()

            # Validation with F1-Score
            model.eval()
            with torch.no_grad():
                val_cat_tensor = torch.LongTensor(X_val_cat)
                val_cont_tensor = torch.FloatTensor(X_val_cont)
                probs = model(val_cat_tensor, val_cont_tensor).numpy()
                preds = (probs > 0.5).astype(int)
                val_f1 = f1_score(y_val, preds)

            # Optuna Pruning
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return val_f1

    # Utilisation de la métrique F1 (maximiser) au lieu de Recall
    study = optuna.create_study(
        direction="maximize",
        study_name="MLP_Embeddings",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    context.log.info(f"✅ Best trial (Validation F1): {study.best_trial.value:.4f}")
    context.log.info(f"📊 Best params: {study.best_params}")

    # Entraîner modèle final avec meilleurs hyperparamètres sur tout le TRAIN (X_train_full)
    best_params = study.best_params

    scaler_final = StandardScaler()
    X_train_cont_final = scaler_final.fit_transform(X_train_full[:, cont_indices])
    X_train_cat_final = X_train_full[:, cat_indices].astype(int)

    scale_weight_full = np.sum(y_train == 0) / np.sum(y_train == 1)

    final_model = MLPEmbeddingClassifier(
        cat_dims=cat_dims_base,
        num_features_cont=len(cont_indices),
        hidden1=best_params["hidden1"],
        hidden2=best_params["hidden2"],
        dropout_rate=best_params["dropout"],
    )
    final_model.scaler = scaler_final
    final_model.cat_indices = cat_indices
    final_model.cont_indices = cont_indices

    optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.BCELoss(reduction="none")
    batch_size = best_params["batch_size"]

    # Training complet (30 epochs)
    context.log.info("🔄 Training final model...")
    for _epoch in range(30):
        final_model.train()
        for i in range(0, len(X_train_full), batch_size):
            batch_cat = torch.LongTensor(X_train_cat_final[i : i + batch_size])
            batch_cont = torch.FloatTensor(X_train_cont_final[i : i + batch_size])
            batch_y = torch.FloatTensor(y_train[i : i + batch_size])
            weights = torch.FloatTensor(
                np.where(y_train[i : i + batch_size] == 1, scale_weight_full, 1.0)
            )

            optimizer.zero_grad()
            output = final_model(batch_cat, batch_cont)
            loss = (criterion(output, batch_y) * weights).mean()
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Prédictions finales (sur l'ensemble de TEST)
    preds = final_model.predict_proba(X_test_full)

    # MLflow logging via resource Dagster
    mlflow_resource = context.resources.mlflow
    mlflow_resource.log_params({f"mlp_{k}": v for k, v in best_params.items()})

    y_pred = (preds[:, 1] > 0.5).astype(int)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, preds[:, 1])
    auc_val = auc_metric([0, 1], [recall, precision])

    mlflow_resource.log_metric("recall", recall)
    mlflow_resource.log_metric("precision", precision)
    mlflow_resource.log_metric("f1", f1)
    mlflow_resource.log_metric("auc", auc_val)
    mlflow_resource.log_metric("auc_roc", auc_roc)

    signature = infer_signature(X_train_full, preds)
    model_info = mlflow.pytorch.log_model(
        pytorch_model=final_model,
        artifact_path="model",
        registered_model_name=registered_model_name,
        signature=signature,
    )
    mlflow_resource.log_param("mlp_registered_model_name", registered_model_name)
    mlflow_resource.log_param("mlp_registered_model_uri", f"models:/{registered_model_name}/latest")
    mlflow_resource.log_param("mlp_registered_model_source_uri", model_info.model_uri)

    save_model(final_model, "mlp_model.pkl")
    mlflow_resource.log_artifact("mlp_model.pkl")

    save_predictions(preds, "mlp_preds.csv")
    mlflow_resource.log_artifact("mlp_preds.csv")

    mlp_metrics = {
        "registered_model_name": registered_model_name,
        "model_uri": f"models:/{registered_model_name}/latest",
        "model_path": "mlp_model.pkl",
        "preds_path": "mlp_preds.csv",
        "auc_roc": auc_roc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }
    with open("mlp_metrics.json", "w") as f:
        json.dump(mlp_metrics, f)
    mlflow_resource.log_artifact("mlp_metrics.json")

    return mlp_metrics
