"""Utilitaires pour assets ML: chargement, sauvegarde, prédiction."""
import joblib
import pandas as pd

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def save_predictions(preds, path):
    pd.DataFrame(preds).to_csv(path, index=False)

def load_predictions(path):
    return pd.read_csv(path)
