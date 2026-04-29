"""跑 TabPFN-v2 vs XGBoost 5-fold CV 对比。

复用 cache_val/ 里已有的 evidence + cls_scores，避开重跑分割集成。

注：TabPFN-2.5 / 2.6 需要 priorlabs.ai 的 TABPFN_TOKEN 才能下权重。
若 TABPFN_TOKEN 未设置，自动回退到 v2（公开 ckpt，不需 token）。
若已设置则跑 v2.5。

用法:
    cd /home/young/TFI
    # 不需要 token 的快速验证 (v2)
    python tools/run_tabpfn_eval.py
    # 拿到 token 后跑 v2.5
    export TABPFN_TOKEN="..."
    python tools/run_tabpfn_eval.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibrator import _build_estimator, find_best_threshold, _brier_logloss_auc
from evidence import FEATURE_NAMES_WITH_CLS, evidence_to_features


def collect_gt_labels(val_dir: Path) -> dict:
    out = {}
    for f in os.listdir(val_dir / "Black" / "Image"):
        out[f] = 1
    for f in os.listdir(val_dir / "White" / "Image"):
        out[f] = 0
    return out


def build_xy() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    val_dir = ROOT / "data" / "raw" / "val"
    stage2 = json.load(open(ROOT / "cache_val" / "stage2_results.json"))
    cls_scores = json.load(open(ROOT / "cache_val" / "cls_scores.json"))
    gt = collect_gt_labels(val_dir)

    names = sorted(stage2.keys())
    X, y, used = [], [], []
    for n in names:
        if n not in gt:
            continue
        ev = stage2[n]["evidence"]
        cls = cls_scores.get(n, {"mean": 0.5, "std": 0.0})
        feats = evidence_to_features(ev, cls_score=cls["mean"], cls_score_std=cls["std"])
        X.append(feats); y.append(gt[n]); used.append(n)
    return np.asarray(X, np.float32), np.asarray(y, np.int32), used


def make_tabpfn(version: str, seed: int):
    """version='v2' or 'v2.5'.  v2 走本地 ckpt 避开 HF 镜像超时。"""
    from tabpfn import TabPFNClassifier
    if version == "v2":
        ckpt = Path.home() / ".cache" / "tabpfn" / "tabpfn-v2-classifier.ckpt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"v2 ckpt not found at {ckpt}. Download first:\n"
                f"  curl -L -o {ckpt} "
                f"https://storage.googleapis.com/tabpfn-v2-model-files/05152025/tabpfn-v2-classifier.ckpt"
            )
        return TabPFNClassifier(model_path=str(ckpt), n_estimators=8,
                                random_state=seed, ignore_pretraining_limits=True)
    elif version == "v2.5":
        from tabpfn.constants import ModelVersion
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V2_5, n_estimators=8, random_state=seed,
        )
    raise ValueError(version)


def cv_eval(name: str, X, y, build_fn, n_splits=5, seed=42):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), -1.0, dtype=np.float64)
    per_fold_f1, fold_times = [], []
    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        t0 = time.time()
        est = build_fn(seed + k)
        est.fit(X[tr], y[tr])
        p_va = est.predict_proba(X[va])[:, 1]
        oof[va] = p_va
        f_t, f_f1 = find_best_threshold(p_va, y[va])
        per_fold_f1.append(f_f1)
        fold_times.append(time.time() - t0)
        print(f"  [{name}] fold {k}/{n_splits}  f1={f_f1:.4f} (t={f_t:.2f})  {time.time()-t0:.1f}s")
    best_t, oof_f1 = find_best_threshold(oof, y)
    brier, ll, auc = _brier_logloss_auc(oof, y)
    return {
        "backend": name,
        "oof_f1": oof_f1, "auc": auc, "brier": brier, "logloss": ll,
        "best_t": best_t,
        "per_fold_f1": per_fold_f1,
        "fold_time_s": fold_times,
    }


def main():
    X, y, names = build_xy()
    print(f"feature matrix: X={X.shape}  y={y.shape}  pos={int(y.sum())}/{len(y)}")
    print(f"feature names ({len(FEATURE_NAMES_WITH_CLS)}): {FEATURE_NAMES_WITH_CLS}")

    has_token = bool(os.environ.get("TABPFN_TOKEN", "").strip())
    tabpfn_version = "v2.5" if has_token else "v2"
    print(f"\n[tabpfn] using {tabpfn_version} (TABPFN_TOKEN {'set' if has_token else 'NOT set, fallback to v2'})")

    print(f"\n{'='*60}\n[run] tabpfn-{tabpfn_version}\n{'='*60}")
    rep_tab = cv_eval(f"tabpfn-{tabpfn_version}", X, y,
                      lambda s: make_tabpfn(tabpfn_version, s))

    print(f"\n{'='*60}\n[run] xgb (baseline)\n{'='*60}")
    def build_xgb(seed):
        est = _build_estimator("xgb", FEATURE_NAMES_WITH_CLS, seed=seed)
        n_pos = max(int(y.sum()), 1); n_neg = max(len(y) - n_pos, 1)
        est.set_params(scale_pos_weight=n_neg / n_pos)
        from sklearn.calibration import CalibratedClassifierCV
        return CalibratedClassifierCV(est, method="isotonic", cv=3)
    rep_xgb = cv_eval("xgb", X, y, build_xgb)

    md = ["| backend | OOF F1 | AUC | Brier↓ | LogLoss↓ | best_t | per-fold F1 (mean±std) | fold time |",
          "|---|---:|---:|---:|---:|---:|---|---:|"]
    for r in (rep_tab, rep_xgb):
        m = float(np.mean(r["per_fold_f1"])); s = float(np.std(r["per_fold_f1"]))
        ft = float(np.mean(r["fold_time_s"]))
        md.append(f"| `{r['backend']}` | **{r['oof_f1']:.4f}** | {r['auc']:.4f} | "
                  f"{r['brier']:.4f} | {r['logloss']:.4f} | {r['best_t']:.2f} | "
                  f"{m:.4f} ± {s:.4f} | {ft:.1f}s |")
    print("\n" + "\n".join(md) + "\n")

    out_path = ROOT / "checkpoints" / "calibrator" / "compare_tabpfn.md"
    header = (
        f"# Calibrator backend comparison: TabPFN-{tabpfn_version} vs XGBoost (5-fold CV)\n\n"
        f"n={len(y)}, positive={int(y.sum())}, features={len(FEATURE_NAMES_WITH_CLS)}\n\n"
    )
    out_path.write_text(header + "\n".join(md) + "\n", encoding="utf-8")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
