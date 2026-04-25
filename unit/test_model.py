import os
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "05_model_input")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "06_models", "model.pkl")
MODEL_EXISTS = os.path.exists(MODEL_PATH)


def load_data():
    raw = joblib.load(MODEL_PATH)
    if isinstance(raw, dict):
        model = raw.get("model") or raw.get("best_model") or list(raw.values())[0]
    else:
        model = raw
    X_test = pd.read_csv(os.path.join(BASE, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(BASE, "y_test.csv")).values.flatten()
    return model, X_test, y_test


@pytest.mark.skipif(not MODEL_EXISTS, reason="model.pkl not available in CI")
def test_invariance_price():
    model, X_test, _ = load_data()
    if "price" not in X_test.columns:
        return
    X_price = X_test[X_test["price"] > 1].copy()
    X_plus = X_price.copy()
    X_plus["price"] = X_plus["price"] + 1
    X_minus = X_price.copy()
    X_minus["price"] = X_minus["price"] - 1
    y_plus = model.predict_proba(X_plus)[:, 1]
    y_minus = model.predict_proba(X_minus)[:, 1]
    abs_delta = np.abs(y_plus - y_minus)
    mean_delta = abs_delta.mean()
    assert mean_delta < 0.30, f"Variation moyenne trop élevée : {mean_delta:.3f}"


@pytest.mark.skipif(not MODEL_EXISTS, reason="model.pkl not available in CI")
def test_directional_duration():
    model, X_test, _ = load_data()
    if "duration" not in X_test.columns:
        return
    X_more = X_test.copy()
    X_more["duration"] = X_more["duration"] + 60
    prob_orig = model.predict_proba(X_test)[:, 1].mean()
    prob_more = model.predict_proba(X_more)[:, 1].mean()
    assert prob_more >= prob_orig - 0.05, (
        f"Augmenter duration devrait augmenter la proba: {prob_orig:.3f} -> {prob_more:.3f}"
    )


@pytest.mark.skipif(not MODEL_EXISTS, reason="model.pkl not available in CI")
def test_prototypes():
    model, X_test, _ = load_data()
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(X_test)
    centroids = pd.DataFrame(data=kmeans.cluster_centers_, columns=X_test.columns)
    probas = model.predict_proba(centroids)[:, 1]
    assert len(probas) == 5
    assert all(0 <= p <= 1 for p in probas), "Les probabilités doivent être entre 0 et 1"
