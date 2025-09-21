import json, random
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import torch

from lightgbm.basic import LightGBMError
from utils.metrics_sepsis import evaluate_sepsis_from_lists

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _featurize_window(w: np.ndarray) -> np.ndarray:
    """
    Summarize window w:[L,F] into fixed vector

    Features per raw feature:
      - last  : last value in the window
      - mean  : average over the window
      - min   : min over the window
      - max   : max over the window
      - slope : (last - first) / L
    Plus:
      - L     : scalar window length

    Return: vector of length(5*F + 1).
    """
    w = w.astype(np.float32, copy=False)
    L, F = w.shape

    last  = w[-1, :]
    mean  = w.mean(axis=0)
    mn    = w.min(axis=0)
    mx    = w.max(axis=0)
    first = w[0, :]
    slope = (last - first) / max(1, L)

    L_feat = np.array([float(L)], dtype=np.float32)

    feats = np.concatenate([last, mean, mn, mx, slope, L_feat]).astype(np.float32)
    return feats


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _build_windows_variable(
    dataset,
    W: int = 12,
    horizon: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    X_rows, y_rows, pid_rows, t_rows = [], [], [], []

    for pid in range(len(dataset)):
        X_tf, y_t = dataset[pid]

        X_tf = _to_numpy(X_tf)
        y_t  = _to_numpy(y_t)

        if X_tf.ndim == 3 and X_tf.shape[0] == 1:
            X_tf = X_tf[0]
        else:
            X_tf = np.squeeze(X_tf)

        if y_t.ndim == 2 and y_t.shape[0] == 1:
            y_t = y_t[0]
        else:
            y_t = np.squeeze(y_t)

        if X_tf.ndim != 2:
            raise ValueError(f"Expected X_tf [T,F], got {X_tf.shape} for pid={pid}")
        if y_t.ndim != 1:
            raise ValueError(f"Expected y_t [T], got {y_t.shape} for pid={pid}")

        X_tf = X_tf.astype(np.float32, copy=False)
        y_t  = y_t.astype(np.int32,   copy=False)

        T, F = X_tf.shape
        stop = max(0, T - horizon)
        for t in range(stop):
            s = max(0, t - W + 1)   
            w = X_tf[s:t+1]         # [L, F]
            feats = _featurize_window(w)  # [D = 5*F + 1]
            X_rows.append(feats)
            y_rows.append(int(y_t[t + horizon]))
            pid_rows.append(pid)
            t_rows.append(t)

    X = np.vstack(X_rows).astype(np.float32) if X_rows else np.zeros((0, 1), dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.int32)
    pid_rows = np.asarray(pid_rows, dtype=np.int64)
    t_rows   = np.asarray(t_rows,   dtype=np.int64)
    return X, y, pid_rows, t_rows


def _scale_pos_weight(y: np.ndarray) -> float:
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return float(neg) / float(max(pos, 1))


def _reconstruct_sequences(pid_rows, t_rows, y_rows, p_rows, threshold: float):
    from collections import defaultdict
    lab_seq = defaultdict(list)
    prob_seq = defaultdict(list)
    pred_seq = defaultdict(list)

    for pid, t, y, p in zip(pid_rows, t_rows, y_rows, p_rows):
        lab_seq[int(pid)].append((int(t), int(y)))
        prob_seq[int(pid)].append((int(t), float(p)))
        pred_seq[int(pid)].append((int(t), int(p >= threshold)))

    cohort_labels, cohort_probs, cohort_preds = [], [], []
    for pid in prob_seq.keys():
        l = [y for t, y in sorted(lab_seq[pid])]
        q = [p for t, p in sorted(prob_seq[pid])]
        b = [z for t, z in sorted(pred_seq[pid])]
        cohort_labels.append(np.array(l, dtype=int))
        cohort_probs.append(np.array(q, dtype=float))
        cohort_preds.append(np.array(b, dtype=int))
    return cohort_labels, cohort_probs, cohort_preds


def _sweep_threshold_for_utility(pid, t, y, p, thresholds=None):
    p = np.asarray(p, dtype=float)

    if thresholds is None:
        
        base = np.r_[np.linspace(0.01, 0.20, 20),
                     np.linspace(0.20, 0.50, 16),
                     np.linspace(0.50, 0.80, 7)]
        
        qs = np.quantile(p, [0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99, 0.995, 0.999])
        thresholds = np.unique(np.clip(np.r_[base, qs], 0.0, 1.0))

    best_thr, best_util = 0.5, -1e9
    for thr in thresholds:
        labs, probs, preds = _reconstruct_sequences(pid, t, y, p, threshold=float(thr))
        auroc, auprc, acc, f1, util = evaluate_sepsis_from_lists(labs, preds, probs)
        if util > best_util:
            best_util, best_thr = util, float(thr)
    return best_thr, best_util


def run_tabular_baselines(
    train_dataset,
    test_dataset,
    output_dir: str,
    window: int = 12,
    run_lr: bool = True,
    run_xgb: bool = True,
    run_lgbm: bool = True,
    threshold: float = 0.5,
    horizon: int = 0,
    tune_threshold: bool = True,
    val_fraction: float = 0.1,
    seed: int = 1,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    print(f"[baselines] Building windows (W={window}, H={horizon}) on TRAIN...")
    X_all, y_all, pid_all, t_all = _build_windows_variable(train_dataset, W=window, horizon=horizon)
    if X_all.shape[0] == 0:
        print("[baselines] No rows constructed — check your datasets.")
        return
    print(f"[baselines] Feature dim D = {X_all.shape[1]}")

    unique_pids = list(set(pid_all.tolist()))
    rng.shuffle(unique_pids)
    n_val = int(round(val_fraction * len(unique_pids)))
    val_pids = set(unique_pids[:n_val])
    trn_mask = np.array([pid not in val_pids for pid in pid_all], dtype=bool)
    val_mask = ~trn_mask

    X_tr, y_tr = X_all[trn_mask], y_all[trn_mask]
    X_val, y_val = (X_all[val_mask], y_all[val_mask]) if np.any(val_mask) else (None, None)
    pid_val, t_val = (pid_all[val_mask], t_all[val_mask]) if np.any(val_mask) else (None, None)

    print(f"[baselines] Train rows: {X_tr.shape[0]:,}  Val rows: {0 if X_val is None else X_val.shape[0]:,}")

    print(f"[baselines] Building windows on TEST...")
    X_te, y_te, pid_te, t_te = _build_windows_variable(test_dataset, W=window, horizon=horizon)
    print(f"[baselines] Test rows: {X_te.shape[0]:,}")

    results: Dict[str, Any] = {}

    if run_lr:
        print("\n[LR] Training Logistic Regression (saga)...")
        scaler = StandardScaler(with_mean=True, with_std=True).fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)
        X_val_s = scaler.transform(X_val) if X_val is not None else None

        lr = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='saga',
            max_iter=300,
            class_weight='balanced',
            n_jobs=-1
        )
        lr.fit(X_tr_s, y_tr)
        print("[LR] Trained.")

        thr_lr = threshold
        if tune_threshold and X_val_s is not None and X_val_s.shape[0] > 0:
            p_val = lr.predict_proba(X_val_s)[:, 1]
            thr_lr, best_util = _sweep_threshold_for_utility(pid_val, t_val, y_val, p_val)
            print(f"[LR] Tuned threshold={thr_lr:.3f} (best val utility={best_util:.4f})")
        else:
            print(f"[LR] Using fixed threshold={thr_lr:.3f}")

        p_te = lr.predict_proba(X_te_s)[:, 1]
        labs, probs, preds = _reconstruct_sequences(pid_te, t_te, y_te, p_te, threshold=thr_lr)
        auroc, auprc, acc, f1, util = evaluate_sepsis_from_lists(labs, preds, probs)
        print(f"[LR/test] AUROC={auroc:.4f} AUPRC={auprc:.4f} Acc={acc:.4f} F1={f1:.4f} Utility={util:.4f}")
        results["logreg"] = {"threshold": thr_lr, "auroc": auroc, "auprc": auprc, "acc": acc, "f1": f1, "utility": util}


    if run_xgb:
        if xgb is None:
            print("[XGB] xgboost is not installed; skipping.")
        else:
            print("\n[XGB] Training XGBoost (gpu_hist, fixed trees)...")
            spw = _scale_pos_weight(y_tr)
            params = dict(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.07,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                min_child_weight=5,
                objective='binary:logistic',
                scale_pos_weight=spw,
                n_jobs=-1,
                tree_method='hist',   
                device='cuda',        
                eval_metric='aucpr',
            )
            xgb_clf = xgb.XGBClassifier(**params)

            if X_val is not None and X_val.shape[0] > 0:
                xgb_clf.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=True            
                )
            else:
                xgb_clf.fit(X_tr, y_tr)

            print(f"[XGB] Trained. scale_pos_weight={spw:.2f}")

            
            p_val = None
            if X_val is not None and X_val.shape[0] > 0:
                p_val = xgb_clf.predict_proba(X_val)[:, 1]

            thr_xgb = threshold
            if tune_threshold and p_val is not None:
                thr_xgb, best_util = _sweep_threshold_for_utility(pid_val, t_val, y_val, p_val)
                print(f"[XGB] Tuned threshold={thr_xgb:.3f} (best val utility={best_util:.4f})")
            else:
                print(f"[XGB] Using fixed threshold={thr_xgb:.3f}")

            p_te = xgb_clf.predict_proba(X_te)[:, 1]
            labs, probs, preds = _reconstruct_sequences(pid_te, t_te, y_te, p_te, threshold=thr_xgb)
            auroc, auprc, acc, f1, util = evaluate_sepsis_from_lists(labs, preds, probs)
            print(f"[XGB/test] AUROC={auroc:.4f} AUPRC={auprc:.4f} Acc={acc:.4f} F1={f1:.4f} Utility={util:.4f}")
            results["xgboost"] = {"threshold": thr_xgb, "auroc": auroc, "auprc": auprc, "acc": acc, "f1": f1, "utility": util}


    if run_lgbm:
        if lgb is None:
            print("[LGBM] lightgbm is not installed; skipping.")
        else:
            print("\n[LGBM] Training LightGBM (gpu + early stopping)...")
            spw = _scale_pos_weight(y_tr)
            params = dict(
                n_estimators=600,
                learning_rate=0.07,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_lambda=1.0,
                objective='binary',
                n_jobs=-1,
                device_type='gpu',
                is_unbalance=True  
               
            )
            lgbm = lgb.LGBMClassifier(**params)

            def _fit_lgbm(model, X_tr, y_tr, X_val, y_val):
                if X_val is not None and X_val.shape[0] > 0:
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        eval_metric='average_precision',
                        callbacks=[
                            lgb.early_stopping(50),
                            lgb.log_evaluation(50),  
                        ],
                    )
                else:
                    model.fit(X_tr, y_tr)

            try:
                _fit_lgbm(lgbm, X_tr, y_tr, X_val, y_val)
            except LightGBMError as e:
                print(f"[LGBM] GPU failed ({e}). Falling back to CPU.")
                # Recreate on CPU and refit
                params['device_type'] = 'cpu'
                lgbm = lgb.LGBMClassifier(**params)
                _fit_lgbm(lgbm, X_tr, y_tr, X_val, y_val)

            
            thr_lgb = threshold
            if tune_threshold and X_val is not None and X_val.shape[0] > 0:
                p_val = lgbm.predict_proba(X_val, num_iteration=getattr(lgbm, "best_iteration_", None))[:, 1]
                thr_lgb, best_util = _sweep_threshold_for_utility(pid_val, t_val, y_val, p_val)
                print(f"[LGBM] Tuned threshold={thr_lgb:.3f} (best val utility={best_util:.4f})")
            else:
                print(f"[LGBM] Using fixed threshold={thr_lgb:.3f}")

            p_te = lgbm.predict_proba(X_te, num_iteration=getattr(lgbm, "best_iteration_", None))[:, 1]
            labs, probs, preds = _reconstruct_sequences(pid_te, t_te, y_te, p_te, threshold=thr_lgb)
            auroc, auprc, acc, f1, util = evaluate_sepsis_from_lists(labs, preds, probs)
            print(f"[LGBM/test] AUROC={auroc:.4f} AUPRC={auprc:.4f} Acc={acc:.4f} F1={f1:.4f} Utility={util:.4f}")
            results["lightgbm"] = {"threshold": thr_lgb, "auroc": auroc, "auprc": auprc, "acc": acc, "f1": f1, "utility": util}

    out_json = Path(output_dir) / "baseline_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[baselines] Saved metrics to: {out_json}")
