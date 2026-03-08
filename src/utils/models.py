"""Model persistence utilities."""

import joblib
import numpy as np
import pandas as pd


def save_model(model, path: str) -> None:
    """Save model to disk using joblib."""
    joblib.dump(model, path)


def load_model(path: str):
    """Load model from disk using joblib."""
    return joblib.load(path)


def save_predictions(preds, path: str) -> None:
    """Save predictions/probabilities to CSV."""
    if isinstance(preds, np.ndarray):
        df = pd.DataFrame(preds)
        if df.shape[1] == 1:
            df.columns = ["prediction"]
        elif df.shape[1] == 2:
            df.columns = ["prob_0", "prob_1"]
        df.to_csv(path, index=False)
    else:
        preds.to_csv(path, index=False)
