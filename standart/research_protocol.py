import os
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def mia_features_basic(proba: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(proba, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    y = np.asarray(y_true, dtype=np.int32)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    conf = np.maximum(p, 1 - p)
    ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    cor = ((p >= 0.5).astype(np.int32) == y).astype(np.float64)
    margin = np.abs(p - 0.5)
    return np.column_stack([loss, conf, ent, cor, margin])


def mia_features_enriched(proba: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(proba, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y_true, dtype=np.int32)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    conf = np.maximum(p, 1 - p)
    ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    cor = ((p >= 0.5).astype(np.int32) == y).astype(np.float64)
    margin = np.abs(p - 0.5)
    logit = np.log(p / (1.0 - p))
    p_true = np.where(y == 1, p, 1.0 - p)
    p_false = 1.0 - p_true
    feats = np.column_stack([loss, conf, ent, cor, margin, p, logit, p_true, p_false])
    return np.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6)


def attack_row(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return {
        "attack": name,
        "auc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _fit_shadow_meta_classifier(
    X_shadow_raw: np.ndarray,
    y_shadow: np.ndarray,
    model_builder: Callable[[int], tf.keras.Model],
    seed: int,
    n_shadows: int,
    shadow_epochs: int,
    shadow_batch_size: int,
    postprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> GradientBoostingClassifier:
    post_fn = postprocess_fn if postprocess_fn is not None else (lambda p: p)

    shadow_x_all: List[np.ndarray] = []
    shadow_y_all: List[np.ndarray] = []

    for i in range(n_shadows):
        xs_mem, xs_non, ys_mem, ys_non = train_test_split(
            X_shadow_raw,
            y_shadow,
            test_size=0.5,
            random_state=seed + 100 + i,
            stratify=y_shadow,
        )

        scaler = StandardScaler()
        xs_mem_sc = scaler.fit_transform(xs_mem).astype(np.float32)
        xs_non_sc = scaler.transform(xs_non).astype(np.float32)

        xs_mem_seq = xs_mem_sc.reshape(-1, 1, xs_mem_sc.shape[1])
        xs_non_seq = xs_non_sc.reshape(-1, 1, xs_non_sc.shape[1])

        set_seed(seed + 2000 + i)
        tf.keras.backend.clear_session()
        sh_model = model_builder(xs_mem_sc.shape[1])
        sh_model.fit(xs_mem_seq, ys_mem, epochs=shadow_epochs, batch_size=shadow_batch_size, verbose=0)

        ps_mem = post_fn(sh_model.predict(xs_mem_seq, verbose=0).ravel())
        ps_non = post_fn(sh_model.predict(xs_non_seq, verbose=0).ravel())

        fx = np.vstack([mia_features_enriched(ps_mem, ys_mem), mia_features_enriched(ps_non, ys_non)])
        fy = np.concatenate([np.ones(len(ys_mem), dtype=int), np.zeros(len(ys_non), dtype=int)])

        shadow_x_all.append(fx)
        shadow_y_all.append(fy)

    X_meta = np.vstack(shadow_x_all)
    y_meta = np.concatenate(shadow_y_all)

    meta = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        random_state=seed,
    )
    meta.fit(X_meta, y_meta)
    return meta


def evaluate_mia_research_protocol(
    target_model: tf.keras.Model,
    X_train_seq: np.ndarray,
    y_train: np.ndarray,
    X_test_seq: np.ndarray,
    y_test: np.ndarray,
    X_shadow_raw: np.ndarray,
    y_shadow: np.ndarray,
    model_builder: Callable[[int], tf.keras.Model],
    seed_list: List[int],
    n_shadows: int = 10,
    shadow_epochs: int = 35,
    shadow_batch_size: int = 16,
    postprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    post_fn = postprocess_fn if postprocess_fn is not None else (lambda p: p)

    p_mem = post_fn(target_model.predict(X_train_seq, verbose=0).ravel())
    p_non = post_fn(target_model.predict(X_test_seq, verbose=0).ravel())

    X_basic = np.vstack([mia_features_basic(p_mem, y_train), mia_features_basic(p_non, y_test)])
    X_enriched = np.vstack([mia_features_enriched(p_mem, y_train), mia_features_enriched(p_non, y_test)])
    y_attack = np.concatenate([np.ones(len(y_train), dtype=int), np.zeros(len(y_test), dtype=int)])

    meta = _fit_shadow_meta_classifier(
        X_shadow_raw=X_shadow_raw,
        y_shadow=y_shadow,
        model_builder=model_builder,
        seed=seed_list[0],
        n_shadows=n_shadows,
        shadow_epochs=shadow_epochs,
        shadow_batch_size=shadow_batch_size,
        postprocess_fn=post_fn,
    )

    rows = []
    for seed in seed_list:
        Xb_tr, Xb_te, Xe_tr, Xe_te, ya_tr, ya_te = train_test_split(
            X_basic,
            X_enriched,
            y_attack,
            test_size=0.4,
            random_state=seed,
            stratify=y_attack,
        )

        score_tr = -Xb_tr[:, 0]
        score_te = -Xb_te[:, 0]
        cand = np.linspace(score_tr.min(), score_tr.max(), 300)
        best_thr = cand[0]
        best_bal = -1.0
        for t in cand:
            pred = (score_tr >= t).astype(int)
            tpr = ((pred == 1) & (ya_tr == 1)).sum() / max((ya_tr == 1).sum(), 1)
            tnr = ((pred == 0) & (ya_tr == 0)).sum() / max((ya_tr == 0).sum(), 1)
            bal = 0.5 * (tpr + tnr)
            if bal > best_bal:
                best_bal = bal
                best_thr = t
        thr_pred = (score_te >= best_thr).astype(int)
        rows.append({"seed": seed, **attack_row("threshold_loss", ya_te, thr_pred, score_te)})

        lr = LogisticRegression(max_iter=1000, random_state=seed)
        lr.fit(Xb_tr, ya_tr)
        lr_score = lr.predict_proba(Xb_te)[:, 1]
        lr_pred = (lr_score >= 0.5).astype(int)
        rows.append({"seed": seed, **attack_row("logistic", ya_te, lr_pred, lr_score)})

        sh_score = meta.predict_proba(Xe_te)[:, 1]
        sh_pred = (sh_score >= 0.5).astype(int)
        rows.append({"seed": seed, **attack_row("shadow_meta", ya_te, sh_pred, sh_score)})

    per_seed = pd.DataFrame(rows)

    summary = (
        per_seed.groupby("attack")[["auc", "accuracy", "precision", "recall", "f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "attack",
        "auc_mean",
        "auc_std",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
    ]
    summary = summary.sort_values("auc_mean", ascending=False)

    return summary, per_seed


def compare_defense_vs_baseline(
    baseline_summary: pd.DataFrame,
    defense_summary: pd.DataFrame,
    defense_name: str,
) -> pd.DataFrame:
    cmp = baseline_summary[["attack", "auc_mean", "accuracy_mean", "f1_mean"]].merge(
        defense_summary[["attack", "auc_mean", "accuracy_mean", "f1_mean"]],
        on="attack",
        suffixes=("_baseline", f"_{defense_name}"),
    )
    cmp["delta_auc"] = cmp[f"auc_mean_{defense_name}"] - cmp["auc_mean_baseline"]
    cmp["delta_accuracy"] = cmp[f"accuracy_mean_{defense_name}"] - cmp["accuracy_mean_baseline"]
    cmp["delta_f1"] = cmp[f"f1_mean_{defense_name}"] - cmp["f1_mean_baseline"]
    return cmp.sort_values("attack")
