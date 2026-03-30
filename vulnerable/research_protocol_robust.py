import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from research_protocol import attack_row, mia_features_enriched


def rank01(x):
    x = np.asarray(x)
    order = np.argsort(np.argsort(x))
    if len(x) <= 1:
        return np.zeros_like(x, dtype=np.float64)
    return order.astype(np.float64) / float(len(x) - 1)


def normal_logpdf(x, mu, sigma):
    sigma = max(float(sigma), 1e-3)
    z = (x - mu) / sigma
    return -0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(sigma) + z * z)


def tpr_at_fpr(y_true, y_score, target_fpr=0.01):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    idx = np.argsort(-y_score)
    y_true = y_true[idx]
    y_score = y_score[idx]

    p = max((y_true == 1).sum(), 1)
    n = max((y_true == 0).sum(), 1)
    tp = 0
    fp = 0
    best_tpr = 0.0
    for yi in y_true:
        if yi == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / n
        if fpr <= target_fpr:
            best_tpr = tp / p
        else:
            break
    return float(best_tpr)


def _predict_label_blackbox(model, x_2d):
    x_seq = x_2d.reshape(-1, 1, x_2d.shape[1])
    p = model.predict(x_seq, verbose=0).ravel()
    return (p >= 0.5).astype(np.int32)


def _boundary_distance_label_only(model, x, pred_label, seed, max_alpha=2.0, n_dirs=20, n_steps=8):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n_dirs, x.shape[0]))
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)

    alphas = np.geomspace(0.02, max_alpha, num=n_steps)
    candidates = []
    for i in range(n_dirs):
        for a in alphas:
            candidates.append(x + a * d[i])
    candidates = np.asarray(candidates, dtype=np.float32)

    labels = _predict_label_blackbox(model, candidates)
    labels = labels.reshape(n_dirs, n_steps)
    flips = labels != int(pred_label)

    dist = max_alpha * 1.5
    for i in range(n_dirs):
        where = np.where(flips[i])[0]
        if len(where) > 0:
            cand = float(alphas[int(where[0])])
            if cand < dist:
                dist = cand
    return dist


def _build_shadow_artifacts(
    x_shadow_raw,
    y_shadow,
    model_builder,
    n_shadows,
    shadow_epochs,
    shadow_batch_size,
    seed,
    postprocess_fn=None,
):
    post_fn = postprocess_fn if postprocess_fn is not None else (lambda p: p)

    shadow_features = []
    shadow_membership = []
    shadow_cls = []
    shadow_in_scores = {0: [], 1: []}
    shadow_out_scores = {0: [], 1: []}

    for s in range(n_shadows):
        xs_mem, xs_non, ys_mem, ys_non = train_test_split(
            x_shadow_raw,
            y_shadow,
            test_size=0.5,
            random_state=seed + 100 + s,
            stratify=y_shadow,
        )

        sc = StandardScaler()
        xs_mem_sc = sc.fit_transform(xs_mem).astype(np.float32)
        xs_non_sc = sc.transform(xs_non).astype(np.float32)

        xs_mem_seq = xs_mem_sc.reshape(-1, 1, xs_mem_sc.shape[1])
        xs_non_seq = xs_non_sc.reshape(-1, 1, xs_non_sc.shape[1])

        tf.keras.backend.clear_session()
        sh_model = model_builder(xs_mem_sc.shape[1])
        sh_model.fit(xs_mem_seq, ys_mem, epochs=shadow_epochs, batch_size=shadow_batch_size, verbose=0)

        ps_mem = post_fn(sh_model.predict(xs_mem_seq, verbose=0).ravel())
        ps_non = post_fn(sh_model.predict(xs_non_seq, verbose=0).ravel())

        fm = mia_features_enriched(ps_mem, ys_mem)
        fn = mia_features_enriched(ps_non, ys_non)

        shadow_features.append(np.vstack([fm, fn]))
        shadow_membership.append(np.concatenate([np.ones(len(fm), dtype=int), np.zeros(len(fn), dtype=int)]))
        shadow_cls.append(np.concatenate([ys_mem.astype(int), ys_non.astype(int)]))

        p_true_mem = np.where(ys_mem == 1, ps_mem, 1.0 - ps_mem)
        p_true_non = np.where(ys_non == 1, ps_non, 1.0 - ps_non)
        s_mem = np.log(np.clip(p_true_mem, 1e-6, 1 - 1e-6) / np.clip(1.0 - p_true_mem, 1e-6, 1 - 1e-6))
        s_non = np.log(np.clip(p_true_non, 1e-6, 1 - 1e-6) / np.clip(1.0 - p_true_non, 1e-6, 1 - 1e-6))

        for cls in [0, 1]:
            shadow_in_scores[cls].extend(s_mem[ys_mem == cls].tolist())
            shadow_out_scores[cls].extend(s_non[ys_non == cls].tolist())

    x_shadow_attack = np.vstack(shadow_features)
    y_shadow_membership = np.concatenate(shadow_membership)
    y_shadow_cls = np.concatenate(shadow_cls)

    lira_params = {}
    for cls in [0, 1]:
        arr_in = np.asarray(shadow_in_scores[cls], dtype=np.float64)
        arr_out = np.asarray(shadow_out_scores[cls], dtype=np.float64)
        lira_params[cls] = {
            "mu_in": float(arr_in.mean()) if len(arr_in) else 0.0,
            "sd_in": float(arr_in.std(ddof=1)) if len(arr_in) > 1 else 1.0,
            "mu_out": float(arr_out.mean()) if len(arr_out) else 0.0,
            "sd_out": float(arr_out.std(ddof=1)) if len(arr_out) > 1 else 1.0,
        }

    return x_shadow_attack, y_shadow_membership, y_shadow_cls, lira_params


def evaluate_mia_robust_protocol(
    target_model,
    x_train_seq,
    y_train,
    x_test_seq,
    y_test,
    x_shadow_raw,
    y_shadow,
    model_builder,
    seed_list,
    n_shadows=20,
    shadow_epochs=25,
    shadow_batch_size=16,
    boundary_dirs=20,
    boundary_steps=8,
    boundary_max_alpha=2.0,
    postprocess_fn=None,
):
    post_fn = postprocess_fn if postprocess_fn is not None else (lambda p: p)

    # Attack target payload from real model under test.
    p_mem = post_fn(target_model.predict(x_train_seq, verbose=0).ravel())
    p_non = post_fn(target_model.predict(x_test_seq, verbose=0).ravel())

    f_mem = mia_features_enriched(p_mem, y_train)
    f_non = mia_features_enriched(p_non, y_test)

    x_attack_target = np.vstack([f_mem, f_non])
    y_attack_target = np.concatenate([np.ones(len(f_mem), dtype=int), np.zeros(len(f_non), dtype=int)])
    y_attack_target_cls = np.concatenate([y_train.astype(int), y_test.astype(int)])

    x_query_target = np.vstack([x_train_seq.reshape(len(x_train_seq), -1), x_test_seq.reshape(len(x_test_seq), -1)])
    y_pred_target = np.concatenate([(p_mem >= 0.5).astype(np.int32), (p_non >= 0.5).astype(np.int32)])

    boundary_raw = np.zeros(len(x_query_target), dtype=np.float64)
    for i in range(len(x_query_target)):
        boundary_raw[i] = _boundary_distance_label_only(
            target_model,
            x_query_target[i],
            y_pred_target[i],
            seed=seed_list[0] + 9000 + i,
            max_alpha=boundary_max_alpha,
            n_dirs=boundary_dirs,
            n_steps=boundary_steps,
        )
    score_boundary_all = rank01(boundary_raw)

    x_shadow_attack, y_shadow_membership, y_shadow_cls, lira_params = _build_shadow_artifacts(
        x_shadow_raw=x_shadow_raw,
        y_shadow=y_shadow,
        model_builder=model_builder,
        n_shadows=n_shadows,
        shadow_epochs=shadow_epochs,
        shadow_batch_size=shadow_batch_size,
        seed=seed_list[0],
        postprocess_fn=post_fn,
    )

    rows_seed = []

    for s_eval in seed_list:
        xa_tr_full, xa_te, ya_tr_full, ya_te, yc_tr_full, yc_te, sb_tr_full, sb_te = train_test_split(
            x_attack_target,
            y_attack_target,
            y_attack_target_cls,
            score_boundary_all,
            test_size=0.4,
            random_state=s_eval,
            stratify=y_attack_target,
        )

        xa_tr, xa_val, ya_tr, ya_val, yc_tr, yc_val, sb_tr, sb_val = train_test_split(
            xa_tr_full,
            ya_tr_full,
            yc_tr_full,
            sb_tr_full,
            test_size=0.25,
            random_state=s_eval + 99,
            stratify=ya_tr_full,
        )

        x_meta_train = np.vstack([x_shadow_attack, xa_tr])
        y_meta_train = np.concatenate([y_shadow_membership, ya_tr])
        yc_meta_train = np.concatenate([y_shadow_cls, yc_tr])

        models_by_cls = {}
        for cls in [0, 1]:
            msk = yc_meta_train == cls
            if msk.sum() >= 20 and len(np.unique(y_meta_train[msk])) == 2:
                m_cls = GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=seed_list[0] + cls + s_eval,
                )
                m_cls.fit(x_meta_train[msk], y_meta_train[msk])
                models_by_cls[cls] = m_cls

        global_meta = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=seed_list[0] + s_eval,
        )
        global_meta.fit(x_meta_train, y_meta_train)

        def score_meta_fn(xa, yc):
            out = np.zeros(len(xa), dtype=np.float64)
            for i in range(len(xa)):
                cls = int(yc[i])
                model_i = models_by_cls.get(cls, global_meta)
                out[i] = model_i.predict_proba(xa[i:i + 1])[:, 1][0]
            return out

        def score_lira_fn(xa, yc):
            p_true = np.clip(xa[:, 7], 1e-6, 1 - 1e-6)
            s_vals = np.log(p_true / (1.0 - p_true))
            out = np.zeros(len(xa), dtype=np.float64)
            for i in range(len(xa)):
                prm = lira_params[int(yc[i])]
                li = normal_logpdf(s_vals[i], prm["mu_in"], prm["sd_in"])
                lo = normal_logpdf(s_vals[i], prm["mu_out"], prm["sd_out"])
                out[i] = li - lo
            return out

        score_meta_val = score_meta_fn(xa_val, yc_val)
        score_lira_val = score_lira_fn(xa_val, yc_val)
        score_meta_te = score_meta_fn(xa_te, yc_te)
        score_lira_te = score_lira_fn(xa_te, yc_te)

        candidates = {
            "meta_only": rank01(score_meta_val),
            "lira_only": rank01(score_lira_val),
            "boundary_only": rank01(sb_val),
            "fusion_0.45_0.25_0.30": 0.45 * rank01(score_meta_val) + 0.25 * rank01(score_lira_val) + 0.30 * rank01(sb_val),
            "fusion_0.35_0.45_0.20": 0.35 * rank01(score_meta_val) + 0.45 * rank01(score_lira_val) + 0.20 * rank01(sb_val),
            "fusion_0.30_0.30_0.40": 0.30 * rank01(score_meta_val) + 0.30 * rank01(score_lira_val) + 0.40 * rank01(sb_val),
        }

        best_name = None
        best_auc_val = -1.0
        for name, sc in candidates.items():
            a = -1.0
            try:
                a = roc_auc_score(ya_val, sc)
            except Exception:
                pass
            if a > best_auc_val:
                best_auc_val = a
                best_name = name

        if best_name == "meta_only":
            score_fusion = rank01(score_meta_te)
        elif best_name == "lira_only":
            score_fusion = rank01(score_lira_te)
        elif best_name == "boundary_only":
            score_fusion = rank01(sb_te)
        elif best_name == "fusion_0.45_0.25_0.30":
            score_fusion = 0.45 * rank01(score_meta_te) + 0.25 * rank01(score_lira_te) + 0.30 * rank01(sb_te)
        elif best_name == "fusion_0.35_0.45_0.20":
            score_fusion = 0.35 * rank01(score_meta_te) + 0.45 * rank01(score_lira_te) + 0.20 * rank01(sb_te)
        else:
            score_fusion = 0.30 * rank01(score_meta_te) + 0.30 * rank01(score_lira_te) + 0.40 * rank01(sb_te)

        pred_shadow = (score_fusion >= 0.5).astype(int)

        row = attack_row("shadow_meta", ya_te, pred_shadow, score_fusion)
        row["seed"] = s_eval
        row["selected_scorer"] = best_name
        row["selected_val_auc"] = float(best_auc_val)
        row["auc_meta_only"] = float(np.nan)
        row["auc_lira_only"] = float(np.nan)
        row["auc_boundary_only"] = float(np.nan)
        try:
            row["auc_meta_only"] = roc_auc_score(ya_te, score_meta_te)
            row["auc_lira_only"] = roc_auc_score(ya_te, score_lira_te)
            row["auc_boundary_only"] = roc_auc_score(ya_te, sb_te)
        except Exception:
            pass
        row["tpr_at_1pct_fpr"] = tpr_at_fpr(ya_te, score_fusion, 0.01)
        row["tpr_at_5pct_fpr"] = tpr_at_fpr(ya_te, score_fusion, 0.05)
        rows_seed.append(row)

    shadow_seed_df = pd.DataFrame(rows_seed)

    shadow_summary = {
        "attack": "shadow_meta",
        "auc_mean": float(shadow_seed_df["auc"].mean()),
        "auc_std": float(shadow_seed_df["auc"].std(ddof=1)) if len(shadow_seed_df) > 1 else 0.0,
        "accuracy_mean": float(shadow_seed_df["accuracy"].mean()),
        "accuracy_std": float(shadow_seed_df["accuracy"].std(ddof=1)) if len(shadow_seed_df) > 1 else 0.0,
        "precision_mean": float(shadow_seed_df["precision"].mean()),
        "precision_std": float(shadow_seed_df["precision"].std(ddof=1)) if len(shadow_seed_df) > 1 else 0.0,
        "recall_mean": float(shadow_seed_df["recall"].mean()),
        "recall_std": float(shadow_seed_df["recall"].std(ddof=1)) if len(shadow_seed_df) > 1 else 0.0,
        "f1_mean": float(shadow_seed_df["f1"].mean()),
        "f1_std": float(shadow_seed_df["f1"].std(ddof=1)) if len(shadow_seed_df) > 1 else 0.0,
    }

    summary = pd.DataFrame([shadow_summary])
    return summary, shadow_seed_df
