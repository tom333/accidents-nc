from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import torch

from src.assets.gold.blending_utils import BlendingEnsembleWrapper


def test_blending_ensemble_wrapper():
    mock_cat = MagicMock()
    mock_cat.predict_proba.return_value = np.array([[0.8, 0.2], [0.1, 0.9]])

    mock_xgb = MagicMock()
    mock_xgb.predict_proba.return_value = np.array([[0.7, 0.3], [0.2, 0.8]])

    mock_mlp = MagicMock()
    # Mock PyTorch model execution
    # Return shape should be (N,) for N samples
    mock_mlp.return_value = torch.tensor([0.4, 0.7])

    wrapper = BlendingEnsembleWrapper(
        cat_model=mock_cat,
        xgb_model=mock_xgb,
        mlp_model=mock_mlp,
        mlp_weights=(0.4, 0.4, 0.2),
        threshold=0.5,
    )

    X_dummy = pd.DataFrame({"geo_cluster": [0, 1], "num_feature": [1.0, 2.0]})

    probas = wrapper.predict_proba(X_dummy)

    # shape should be (2, 2)
    assert probas.shape == (2, 2)

    # Sample 1: 0.2*0.4 + 0.3*0.4 + 0.4*0.2 = 0.28
    # Sample 2: 0.9*0.4 + 0.8*0.4 + 0.7*0.2 = 0.82
    np.testing.assert_allclose(probas[:, 1], np.array([0.28, 0.82]))
    np.testing.assert_allclose(probas[:, 0], np.array([0.72, 0.18]))

    preds = wrapper.predict(X_dummy)

    assert preds.shape == (2,)
    # By threshold 0.5: 0.28 < 0.5 -> 0, 0.82 >= 0.5 -> 1
    np.testing.assert_array_equal(preds, np.array([0, 1]))
