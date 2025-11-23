#!/usr/bin/env python
"""

Train a logistic regression model on RRF–Savant meta-state features using
OmegaReflection_log.jsonl, and export:

  - logreg_rrf_savant.joblib
  - config.json

Features (dimension = 15):
  [phi, omega, coherence, S_RRF, C_RRF, E_H, dominant_frequency,
   one-hot over Phi nodes (up to 8 distinct nodes)]

Labels:
  - By default: 3 zones derived from the RRF resonant score R_res
    (exploratory / resonant / optimal via 33% / 66% quantiles).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------------

def load_omega_log(path: Path) -> List[Dict]:
    """Load OmegaReflection_log.jsonl into a list of dicts."""
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"No records found in {path}")
    return records


def build_phi_vocab(records: List[Dict], max_nodes: int = 8) -> List[str]:
    """
    Build a Phi-node vocabulary from closest_phi_node.

    Pads with synthetic labels if fewer than max_nodes exist so that the
    final feature dimension is stable (= 7 + max_nodes).
    """
    unique_nodes = sorted({r["closest_phi_node"] for r in records})
    vocab: List[str] = list(unique_nodes)

    # Pad with dummy nodes if needed
    while len(vocab) < max_nodes:
        vocab.append(f"__PAD_{len(vocab)}")

    # If there were more than max_nodes in some larger dataset, you could
    # truncate here; in this small pilot, you’re well under 8.
    return vocab[:max_nodes]


def build_feature_matrix(records: List[Dict], phi_vocab: List[str]) -> np.ndarray:
    """
    Construct the (N, 15) feature matrix:

    base = [phi, omega, coherence, S_RRF, C_RRF, E_H, dominant_frequency]
    plus one-hot for closest_phi_node over phi_vocab (len 8).
    """
    n_phi = len(phi_vocab)
    feats: List[List[float]] = []

    for r in records:
        phi = float(r["phi"])
        omega = float(r["omega"])
        coherence = float(r["coherence"])
        S_RRF = float(r["S_RRF"])
        C_RRF = float(r["C_RRF"])
        E_H = float(r["hamiltonian_energy"])
        nu = float(r["dominant_frequency"])

        base = [phi, omega, coherence, S_RRF, C_RRF, E_H, nu]

        one_hot = [0.0] * n_phi
        node = r["closest_phi_node"]
        if node in phi_vocab:
            one_hot[phi_vocab.index(node)] = 1.0

        feats.append(base + one_hot)

    X = np.asarray(feats, dtype=float)
    if X.shape[1] != 7 + len(phi_vocab):
        raise RuntimeError(
            f"Unexpected feature dimension {X.shape[1]} "
            f"(expected {7 + len(phi_vocab)})"
        )
    return X


# ---------------------------------------------------------------------------
# RRF resonant score & labels
# ---------------------------------------------------------------------------

def compute_rres(record: Dict) -> float:
    """
    Compute the RRF resonant score R_res as in the RRF–Savant meta-state
    definition:

      R_res = 0.5 * Coherence
              + 0.3 * (0.5 * S_RRF + 0.5 * C_RRF)
              + 0.2 * Phi * (1 - |Omega - 0.5|)

    Using:
      Phi   -> record["phi"]
      Omega -> record["omega"]
      S_RRF -> record["S_RRF"]
      C_RRF -> record["C_RRF"]
      Coherence -> record["coherence"]
    """
    Phi = float(record["phi"])
    Omega = float(record["omega"])
    S_RRF = float(record["S_RRF"])
    C_RRF = float(record["C_RRF"])
    coherence = float(record["coherence"])

    rres = (
        0.5 * coherence
        + 0.3 * (0.5 * S_RRF + 0.5 * C_RRF)
        + 0.2 * Phi * (1.0 - abs(Omega - 0.5))
    )
    return float(rres)


def make_labels_from_rres(
    rres: np.ndarray,
    mode: str = "zones"
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Build labels from R_res:

    mode == "binary":
        y = 0 / 1 using median(R_res) threshold.

    mode == "zones" (default):
        Three classes via quantiles:
          y=0: R_res < 33%  -> "exploratory"
          y=1: 33% <= R_res < 66% -> "resonant"
          y=2: R_res >= 66% -> "optimal"
    """
    if mode not in {"binary", "zones"}:
        raise ValueError("mode must be 'binary' or 'zones'")

    if mode == "binary":
        threshold = float(np.median(rres))
        y = (rres >= threshold).astype(int)
        label_map = {0: "low_Rres", 1: "high_Rres"}
        return y, label_map

    # zones
    q33, q66 = np.quantile(rres, [1 / 3, 2 / 3])
    y = np.digitize(rres, [q33, q66])  # 0,1,2
    label_map = {
        0: "exploratory",
        1: "resonant",
        2: "optimal",
    }
    return y, label_map


# ---------------------------------------------------------------------------
# Training & export
# ---------------------------------------------------------------------------

def train_logreg(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict]:
    """
    Train a StandardScaler + (multi-class) LogisticRegression pipeline.
    Returns the pipeline and a dict of eval metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=random_state,
    )

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                max_iter=1000,
                multi_class="multinomial",
            )),
        ]
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return clf, {
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "classification_report": report,
    }


def save_model_and_config(
    model: Pipeline,
    phi_vocab: List[str],
    label_map: Dict[int, str],
    metrics: Dict,
    model_path: Path,
    config_path: Path,
) -> None:
    """Persist the joblib model and a JSON config file."""
    joblib.dump(model, model_path)

    config = {
        "model_type": "logistic_regression",
        "framework": "scikit-learn",
        "input_features": {
            "description": (
                "RRF-Savant meta-state features "
                "(phi, omega, coherence, S_RRF, C_RRF, E_H, dominant_frequency, "
                "one-hot Phi nodes)"
            ),
            "dimension": int(7 + len(phi_vocab)),
            "phi_node_vocab": phi_vocab,
            "feature_order": [
                "phi",
                "omega",
                "coherence",
                "S_RRF",
                "C_RRF",
                "E_H",
                "dominant_frequency",
                "phi_node_one_hot",
            ],
        },
        "labels": {
            "id_to_name": label_map,
            "n_classes": int(len(label_map)),
        },
        "training_metrics": metrics,
        "created_from": "train_rrf_savant_logreg.py",
        "dependencies": [
            "scikit-learn",
            "numpy",
            "joblib",
        ],
    }

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression on RRF–Savant meta-state features."
    )
    parser.add_argument(
        "--omega-log",
        type=Path,
        default=Path("OmegaReflection_log.jsonl"),
        help="Path to OmegaReflection_log.jsonl",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        default="zones",
        choices=["zones", "binary"],
        help="How to derive labels from R_res (default: zones).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("logreg_rrf_savant.joblib"),
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("config.json"),
        help="Where to save the JSON config.",
    )

    args = parser.parse_args()

    # 1) Load data
    records = load_omega_log(args.omega_log)

    # 2) Build Phi vocabulary and features
    phi_vocab = build_phi_vocab(records, max_nodes=8)
    X = build_feature_matrix(records, phi_vocab)

    # 3) Compute R_res and derive labels
    rres = np.asarray([compute_rres(r) for r in records], dtype=float)
    y, label_map = make_labels_from_rres(rres, mode=args.label_mode)

    # 4) Train model
    model, metrics = train_logreg(X, y)

    # 5) Persist artifacts
    save_model_and_config(
        model=model,
        phi_vocab=phi_vocab,
        label_map=label_map,
        metrics=metrics,
        model_path=args.model_path,
        config_path=args.config_path,
    )

    # 6) Human-readable summary
    print(f"Trained LogisticRegression on {X.shape[0]} samples, dim={X.shape[1]}")
    print(f"Classes: {label_map}")
    print("Classification report (test split):")
    from pprint import pprint as _pprint  # noqa: E402
    _pprint(metrics["classification_report"])


if __name__ == "__main__":
    main()
