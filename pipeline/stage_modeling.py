from __future__ import annotations

import copy
import io
from pathlib import Path
from typing import Dict, Tuple

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from optuna.integration import CatBoostPruningCallback, LightGBMPruningCallback
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from .config import GOLD_SCHEMA, PIPELINE_PARAMS, ensure_connection
from .stage_datasets import FEATURE_COLUMNS


def _prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    conn = ensure_connection()
    train_df = conn.execute(f"SELECT * FROM {GOLD_SCHEMA}.train").df()
    test_df = conn.execute(f"SELECT * FROM {GOLD_SCHEMA}.test").df()
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df['target']
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df['target']
    return X_train, X_test, y_train, y_test


class TabNetServingModel:
    def __init__(self, trained_model: TabNetClassifier, init_params: Dict):
        self.model = trained_model
        self.init_params = init_params

    def predict(self, X):
        data = X.values if hasattr(X, 'values') else X
        return self.model.predict(np.asarray(data, dtype=np.float32))

    def predict_proba(self, X):
        data = X.values if hasattr(X, 'values') else X
        return self.model.predict_proba(np.asarray(data, dtype=np.float32))

    def __getstate__(self):
        buffer = io.BytesIO()
        torch.save(self.model.network.state_dict(), buffer)
        return {'init_params': self.init_params, 'state_bytes': buffer.getvalue()}

    def __setstate__(self, state):
        self.init_params = state['init_params']
        buffer = io.BytesIO(state['state_bytes'])
        model = TabNetClassifier(**self.init_params)
        model.network.load_state_dict(torch.load(buffer, map_location='cpu'))
        model.device_name = 'cpu'
        self.model = model


class EmbeddingMLP(nn.Module):
    def __init__(self, cat_dims, cont_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim, embedding_dim=min(16, max(4, dim // 2)))
            for dim in cat_dims
        ])
        emb_dim = sum(emb.embedding_dim for emb in self.embeddings)
        input_dim = emb_dim + cont_dim if emb_dim + cont_dim > 0 else 1
        hidden_dim = max(64, input_dim * 2)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_cat, x_cont):
        if self.embeddings:
            embed_out = [emb(x_cat[:, idx]) for idx, emb in enumerate(self.embeddings)]
            x = torch.cat(embed_out, dim=1)
            if x_cont.shape[1] > 0:
                x = torch.cat([x, x_cont], dim=1)
        else:
            x = x_cont
        return self.network(x)


class EmbeddingNeuralNetWrapper:
    def __init__(self, model, cat_cols, cont_cols, cat_mappings, scaler):
        self.model = model.cpu()
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.cat_mappings = cat_mappings
        self.scaler = scaler

    def _encode_categorical(self, df):
        if not self.cat_cols:
            return np.zeros((len(df), 0), dtype=np.int64)
        encoded = []
        for col in self.cat_cols:
            meta = self.cat_mappings[col]
            mapping = meta['mapping']
            unknown_index = meta['unknown_index']
            encoded.append(df[col].map(mapping).fillna(unknown_index).astype(np.int64).to_numpy())
        return np.stack(encoded, axis=1)

    def _encode_continuous(self, df):
        if not self.cont_cols:
            return np.zeros((len(df), 0), dtype=np.float32)
        values = df[self.cont_cols].to_numpy(dtype=np.float32)
        if self.scaler is not None:
            values = self.scaler.transform(df[self.cont_cols]).astype(np.float32)
        return values

    def _prepare_tensors(self, X):
        cat_array = self._encode_categorical(X)
        cont_array = self._encode_continuous(X)
        cat_tensor = torch.tensor(cat_array, dtype=torch.long)
        cont_tensor = torch.tensor(cont_array, dtype=torch.float32)
        return cat_tensor, cont_tensor

    def predict_proba(self, X):
        if not hasattr(X, 'values'):
            raise ValueError("Le modèle attend un DataFrame en entrée.")
        cat_tensor, cont_tensor = self._prepare_tensors(X)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(cat_tensor, cont_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        if probs.ndim == 0:
            probs = np.array([probs])
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def __getstate__(self):
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        return {
            'state_bytes': buffer.getvalue(),
            'cat_cols': self.cat_cols,
            'cont_cols': self.cont_cols,
            'cat_mappings': self.cat_mappings,
            'scaler': self.scaler,
        }

    def __setstate__(self, state):
        self.cat_cols = state['cat_cols']
        self.cont_cols = state['cont_cols']
        self.cat_mappings = state['cat_mappings']
        self.scaler = state['scaler']
        cat_dims = [meta['unknown_index'] + 1 for meta in self.cat_mappings.values()]
        model = EmbeddingMLP(cat_dims, len(self.cont_cols))
        buffer = io.BytesIO(state['state_bytes'])
        model.load_state_dict(torch.load(buffer, map_location='cpu'))
        model.eval()
        self.model = model


def _build_categorical_mappings(df, categorical_cols):
    mappings = {}
    dims = []
    for col in categorical_cols:
        uniques = sorted(df[col].dropna().unique().tolist())
        mapping = {val: idx for idx, val in enumerate(uniques)}
        unknown_index = len(mapping)
        mappings[col] = {'mapping': mapping, 'unknown_index': unknown_index}
        dims.append(unknown_index + 1)
    return mappings, dims


def _encode_categorical_matrix(df, mappings):
    if not mappings:
        return np.zeros((len(df), 0), dtype=np.int64)
    encoded = []
    for col, meta in mappings.items():
        mapping = meta['mapping']
        unknown_index = meta['unknown_index']
        encoded.append(df[col].map(mapping).fillna(unknown_index).astype(np.int64).to_numpy())
    return np.stack(encoded, axis=1)


def _train_embedding_model(cat_train, cont_train, y_train_array, cat_valid, cont_valid, y_valid_array, cat_dims, cont_dim, pos_weight, device, max_epochs=50, batch_size=512):
    model = EmbeddingMLP(cat_dims, cont_dim).to(device)
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def to_tensor(array, dtype):
        if array.size == 0:
            return torch.zeros((array.shape[0], 0), dtype=dtype)
        return torch.tensor(array, dtype=dtype)

    train_dataset = TensorDataset(
        to_tensor(cat_train, torch.long),
        to_tensor(cont_train, torch.float32),
        torch.tensor(y_train_array.reshape(-1, 1), dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)

    val_cat = to_tensor(cat_valid, torch.long).to(device)
    val_cont = to_tensor(cont_valid, torch.float32).to(device)
    val_target = torch.tensor(y_valid_array.reshape(-1, 1), dtype=torch.float32, device=device)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float('inf')
    patience_counter = 0
    patience = 6

    for _ in range(max_epochs):
        model.train()
        for batch_cat, batch_cont, batch_target in train_loader:
            batch_cat = batch_cat.to(device)
            batch_cont = batch_cont.to(device)
            batch_target = batch_target.to(device)
            optimizer.zero_grad()
            logits = model(batch_cat, batch_cont)
            loss = criterion(logits, batch_target)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_cat, val_cont)
            val_loss = criterion(val_logits, val_target).item()

        if val_loss + 1e-4 < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    model.to('cpu')
    return model


def run_training() -> dict:
    X_train, X_test, y_train, y_test = _prepare_data()
    params = PIPELINE_PARAMS

    tuning_n_negative = (y_train == 0).sum()
    tuning_n_positive = (y_train == 1).sum()
    tuning_scale_weight = tuning_n_negative / tuning_n_positive

    mlflow.set_experiment("accidents-nc")
    mlflow.start_run(run_name="pipeline-training")
    mlflow.log_params({
        'scale_weight': tuning_scale_weight,
        'n_train_samples': len(y_train),
        'n_test_samples': len(y_test),
        'n_features': len(FEATURE_COLUMNS)
    })

    # Optuna tuning for CatBoost/LightGBM/XGBoost identical to legacy notebook
    def objective_catboost(trial):
        cb_params = {
            'iterations': trial.suggest_int('iterations', 150, 500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'auto_class_weights': 'Balanced',
            'eval_metric': 'AUC',
            'random_state': params.random_state,
            'verbose': False
        }
        model = CatBoostClassifier(**cb_params)
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            callbacks=[CatBoostPruningCallback(trial, 'AUC')],
            early_stopping_rounds=50,
            verbose=False
        )
        preds = model.predict(X_test)
        return recall_score(y_test, preds, pos_label=1)

    study_catboost = optuna.create_study(direction='maximize', study_name='CatBoost', pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study_catboost.optimize(objective_catboost, n_trials=30, show_progress_bar=False)

    def objective_lgbm(trial):
        lgbm_params = {
            'n_estimators': trial.suggest_int('n_estimators', 150, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'is_unbalance': True,
            'metric': 'auc',
            'random_state': params.random_state,
            'verbose': -1
        }
        model = LGBMClassifier(**lgbm_params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[LightGBMPruningCallback(trial, 'auc')])
        preds = model.predict(X_test)
        return recall_score(y_test, preds, pos_label=1)

    study_lgbm = optuna.create_study(direction='maximize', study_name='LightGBM', pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study_lgbm.optimize(objective_lgbm, n_trials=30, show_progress_bar=False)

    def objective_xgb(trial):
        xgb_params = {
            'n_estimators': trial.suggest_int('n_estimators', 150, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': tuning_scale_weight,
            'random_state': params.random_state,
            'verbosity': 0
        }
        model = XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return recall_score(y_test, preds, pos_label=1)

    study_xgb = optuna.create_study(direction='maximize', study_name='XGBoost', pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study_xgb.optimize(objective_xgb, n_trials=30, show_progress_bar=False)

    tuned_models = {}
    tuning_results = []
    model_run_ids = {}
    non_registerable = set()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_array = X_train.to_numpy(dtype=np.float32)
    X_test_array = X_test.to_numpy(dtype=np.float32)

    def _log_metrics(name, model, train_time, params_dict=None):
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        metrics = {
            'recall': recall_score(y_test, preds, pos_label=1),
            'precision': precision_score(y_test, preds, pos_label=1),
            'f1_score': f1_score(y_test, preds, pos_label=1),
            'auc_roc': roc_auc_score(y_test, probs),
            'training_time_seconds': train_time,
        }
        tuned_models[name] = model
        tuning_results.append({
            'Modèle': name,
            'Recall': f"{metrics['recall']:.3f}",
            'Precision': f"{metrics['precision']:.3f}",
            'F1-Score': f"{metrics['f1_score']:.3f}",
            'AUC-ROC': f"{metrics['auc_roc']:.3f}",
            'Temps (s)': 'n/a' if train_time is None else f"{train_time:.0f}",
            '_recall_raw': metrics['recall']
        })
        if params_dict:
            mlflow.log_params({f"{name}.{k}": v for k, v in params_dict.items()})
        mlflow.log_metrics({f"{name}.{k}": v for k, v in metrics.items()})
        return metrics

    # Train tuned tree models
    best_catboost = CatBoostClassifier(**study_catboost.best_params, auto_class_weights='Balanced', eval_metric='AUC', random_state=params.random_state, verbose=False)
    best_catboost.fit(X_train, y_train)
    _log_metrics('CatBoost', best_catboost, None, study_catboost.best_params)

    best_lgbm = LGBMClassifier(**study_lgbm.best_params, is_unbalance=True, metric='auc', random_state=params.random_state, verbose=-1)
    best_lgbm.fit(X_train, y_train)
    _log_metrics('LightGBM', best_lgbm, None, study_lgbm.best_params)

    best_xgb = XGBClassifier(**study_xgb.best_params, scale_pos_weight=tuning_scale_weight, random_state=params.random_state, verbosity=0)
    best_xgb.fit(X_train, y_train)
    _log_metrics('XGBoost', best_xgb, None, study_xgb.best_params)

    logistic_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('log_reg', LogisticRegression(class_weight='balanced', C=0.5, max_iter=2000, solver='lbfgs', random_state=params.random_state))
    ])
    logistic_pipeline.fit(X_train, y_train)
    _log_metrics('LogisticRegression', logistic_pipeline, None, {'C': 0.5, 'solver': 'lbfgs'})

    try:
        tabnet_cat_cols = [col for col in ['atm', 'road_type', 'is_weekend', 'is_holiday', 'school_holidays'] if col in X_train.columns]
        cat_idxs = [X_train.columns.get_loc(col) for col in tabnet_cat_cols]
        cat_dims = [int(X_train[col].nunique()) for col in tabnet_cat_cols]
        tabnet_params = {
            'n_d': 32,
            'n_a': 32,
            'n_steps': 5,
            'gamma': 1.5,
            'cat_idxs': cat_idxs,
            'cat_dims': cat_dims,
            'cat_emb_dim': [min(16, dim + 1) for dim in cat_dims],
            'n_independent': 2,
            'n_shared': 2,
            'seed': params.random_state,
            'verbose': 0
        }
        tabnet_model = TabNetClassifier(**tabnet_params)
        weights = np.where(y_train.to_numpy() == 1, tuning_scale_weight, 1.0)
        tabnet_model.fit(
            X_train_array,
            y_train.to_numpy(),
            eval_set=[(X_test_array, y_test.to_numpy())],
            eval_metric=['auc'],
            max_epochs=200,
            patience=30,
            batch_size=1024,
            virtual_batch_size=128,
            weights=weights,
        )
        tabnet_wrapper = TabNetServingModel(tabnet_model, tabnet_params)
        _log_metrics('TabNet', tabnet_wrapper, None, tabnet_params)
        non_registerable.add('TabNet')
    except Exception as exc:
        print(f"⚠️ TabNet non entraîné : {exc}")

    try:
        neural_cat_cols = [col for col in ['atm', 'road_type'] if col in X_train.columns]
        cat_mappings, cat_dims = _build_categorical_mappings(X_train, neural_cat_cols)
        cat_train_matrix = _encode_categorical_matrix(X_train[neural_cat_cols], cat_mappings) if neural_cat_cols else np.zeros((len(X_train), 0), dtype=np.int64)
        cat_test_matrix = _encode_categorical_matrix(X_test[neural_cat_cols], cat_mappings) if neural_cat_cols else np.zeros((len(X_test), 0), dtype=np.int64)
        cont_cols_nn = [col for col in X_train.columns if col not in neural_cat_cols]
        if cont_cols_nn:
            scaler = StandardScaler()
            cont_train_matrix = scaler.fit_transform(X_train[cont_cols_nn]).astype(np.float32)
            cont_test_matrix = scaler.transform(X_test[cont_cols_nn]).astype(np.float32)
        else:
            scaler = None
            cont_train_matrix = np.zeros((len(X_train), 0), dtype=np.float32)
            cont_test_matrix = np.zeros((len(X_test), 0), dtype=np.float32)
        model = _train_embedding_model(
            cat_train_matrix,
            cont_train_matrix,
            y_train.to_numpy().astype(np.float32),
            cat_test_matrix,
            cont_test_matrix,
            y_test.to_numpy().astype(np.float32),
            cat_dims,
            len(cont_cols_nn),
            tuning_scale_weight,
            device,
        )
        wrapper = EmbeddingNeuralNetWrapper(model, neural_cat_cols, cont_cols_nn, cat_mappings, scaler)
        _log_metrics('NeuralEmbedding', wrapper, None, {'cat_features': len(neural_cat_cols), 'cont_features': len(cont_cols_nn)})
        non_registerable.add('NeuralEmbedding')
    except Exception as exc:
        print(f"⚠️ Réseau de neurones non entraîné : {exc}")

    results_df = pd.DataFrame(tuning_results).sort_values('_recall_raw', ascending=False)
    best_model_name = results_df.iloc[0]['Modèle']
    best_model = tuned_models[best_model_name]
    mlflow.log_metric('best_recall', results_df.iloc[0]['_recall_raw'])
    mlflow.log_param('best_model', best_model_name)
    mlflow.log_table(results_df.drop(columns='_recall_raw'), "comparison_table.json")

    if best_model_name not in non_registerable:
        model_path = Path('accident_model.pkl')
        joblib.dump(best_model, model_path)
        mlflow.sklearn.log_model(best_model, artifact_path="best-model")
    else:
        print(f"ℹ️ Best model {best_model_name} non enregistré (format personnalisé)")

    mlflow.end_run()

    joblib.dump(best_model, 'accident_model.pkl')
    joblib.dump({'name': best_model_name}, 'best_model_info.pkl')

    return {
        'best_model': best_model_name,
        'leaders': results_df.head(5)[['Modèle', 'Recall']].to_dict(orient='records'),
    }
