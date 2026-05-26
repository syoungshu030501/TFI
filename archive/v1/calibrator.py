"""轻量校准器 - 把分割/分类/取证多源信号融合为最终伪造概率。

替代旧版 inference.py 中的硬阈值规则:
    if cls_score < 0.2 and seg_label == 1: flip to 0
    elif cls_score > 0.9 and seg_label == 0: flip to 1

新方案:
    P(forged) = backend(seg_features + cls_features + evidence_features)
    threshold = argmax_F1 on out-of-fold (OOF) probs   ← 不再用训练集 probs

支持的 backend（按 2026-04 状态推荐）:
    backend       依赖             说明
    -------       --------         -----------------------------------------------
    logistic      sklearn          baseline，n=200 d=9 经常打平 boosting；概率最干净
    xgb           xgboost          旧 baseline，自动 isotonic 校准
    lgbm_mono     lightgbm         + 单调约束（seg_conf↑/area↑/anom↑ → P↑）+ isotonic
    tabpfn        tabpfn>=2.5      2026 SOTA，对 ≤10K 样本对 XGB 100% 胜率（非商业）
    ebm           interpret        Explainable Boosting Machine，可解释性 SOTA

跑法:
    >>> from calibrator import Calibrator
    >>> cal = Calibrator.load("checkpoints/calibrator")
    >>> p, label = cal.predict(evidence_dict, cls_score_mean, cls_score_std)
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from evidence import (
    FEATURE_NAMES_BASE,
    FEATURE_NAMES_WITH_CLS,
    evidence_to_features,
)


# ============================================================
#  特征 → 单调方向（仅 lgbm_mono 用）
# ============================================================
# +1 表示「特征值越大，P(forged) 越大」
# -1 表示反向，0 表示无约束
MONOTONE_BY_FEATURE: Dict[str, int] = {
    "seg_confidence": +1,
    "seg_max_prob": +1,
    "total_area_ratio": +1,
    "log_n_regions": +1,
    "largest_area_ratio": +1,
    "ela_anomaly": +1,
    "srm_anomaly": +1,
    "edge_sharpness": 0,        # 真实图也可能很锐利
    "cls_score_mean": +1,
    "cls_score_std": 0,         # std 大表示不确定，不强制单调
}


SUPPORTED_BACKENDS = ("logistic", "xgb", "lgbm_mono", "tabpfn", "ebm")


# ============================================================
#  Calibrator 主类
# ============================================================

class Calibrator:
    """统一接口的校准器，对所有 backend 都暴露 predict_proba/predict。"""

    def __init__(self, model, threshold: float = 0.5,
                 feature_names: Optional[List[str]] = None,
                 backend: str = "logistic"):
        self.model = model
        self.threshold = float(threshold)
        self.feature_names = feature_names or FEATURE_NAMES_WITH_CLS
        self.backend = backend

    def _prep(self, ev: Dict, cls_mean: Optional[float],
              cls_std: Optional[float]) -> np.ndarray:
        feats = evidence_to_features(ev, cls_score=cls_mean, cls_score_std=cls_std)
        if len(feats) != len(self.feature_names):
            feats = feats[: len(self.feature_names)]
        return np.asarray(feats, dtype=np.float32).reshape(1, -1)

    def predict_proba(self, ev: Dict, cls_mean: Optional[float] = None,
                      cls_std: Optional[float] = None) -> float:
        x = self._prep(ev, cls_mean, cls_std)
        return float(self.model.predict_proba(x)[0, 1])

    def predict(self, ev: Dict, cls_mean: Optional[float] = None,
                cls_std: Optional[float] = None) -> Tuple[float, int]:
        p = self.predict_proba(ev, cls_mean, cls_std)
        return p, int(p > self.threshold)

    def save(self, save_dir: str):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "calibrator.pkl", "wb") as f:
            pickle.dump({
                "model": self.model,
                "threshold": self.threshold,
                "feature_names": self.feature_names,
                "backend": self.backend,
            }, f)

    @classmethod
    def load(cls, save_dir: str) -> "Calibrator":
        with open(Path(save_dir) / "calibrator.pkl", "rb") as f:
            blob = pickle.load(f)
        return cls(model=blob["model"], threshold=blob["threshold"],
                   feature_names=blob["feature_names"],
                   backend=blob.get("backend", "logistic"))


# ============================================================
#  阈值搜索 + 评估指标
# ============================================================

def find_best_threshold(probs: np.ndarray, labels: np.ndarray,
                        grid: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """在给定 probs 上扫阈值找最大 F1。"""
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.01)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (probs > t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        if tp == 0:
            f1 = 0.0
        else:
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


def _brier_logloss_auc(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    eps = 1e-7
    p = np.clip(probs, eps, 1 - eps)
    brier = float(np.mean((p - labels) ** 2))
    ll = float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(labels, probs))
    except Exception:
        auc = float("nan")
    return brier, ll, auc


@dataclass
class FoldReport:
    backend: str
    oof_probs: np.ndarray
    labels: np.ndarray
    best_threshold: float
    f1_oof: float
    auc_oof: float
    brier_oof: float
    logloss_oof: float
    per_fold_f1: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "backend": self.backend,
            "n": int(len(self.labels)),
            "best_threshold": self.best_threshold,
            "oof_f1": self.f1_oof,
            "oof_auc": self.auc_oof,
            "oof_brier": self.brier_oof,
            "oof_logloss": self.logloss_oof,
            "per_fold_f1": self.per_fold_f1,
            "per_fold_f1_mean": float(np.mean(self.per_fold_f1)) if self.per_fold_f1 else None,
            "per_fold_f1_std": float(np.std(self.per_fold_f1)) if self.per_fold_f1 else None,
        }


# ============================================================
#  各 backend 的「fit 一次」工厂函数（不做 CV，CV 在外层做）
# ============================================================

def _build_estimator(backend: str, feature_names: List[str], seed: int = 42):
    """返回一个未训练的 sklearn 兼容 estimator。"""
    if backend == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                               random_state=seed),
        )

    if backend == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=1.0,    # 由 caller 按数据再设
            objective="binary:logistic", eval_metric="logloss",
            random_state=seed, n_jobs=4, verbosity=0,
        )

    if backend == "lgbm_mono":
        from lightgbm import LGBMClassifier
        mono = [MONOTONE_BY_FEATURE.get(n, 0) for n in feature_names]
        return LGBMClassifier(
            n_estimators=300, max_depth=-1, num_leaves=15,
            learning_rate=0.05, min_child_samples=5,
            subsample=0.9, colsample_bytree=0.9,
            class_weight="balanced",
            monotone_constraints=mono,
            random_state=seed, n_jobs=4, verbosity=-1,
        )

    if backend == "tabpfn":
        # 2026-04 SOTA，对 ≤10K 样本完胜默认 XGBoost
        # pip install tabpfn>=2.5；首次用会下载 ~140MB 权重
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as e:
            raise ImportError(
                "缺少 tabpfn 包：pip install 'tabpfn>=2.5'  "
                "（注意非商业许可，详见 https://priorlabs.ai/tabpfn）"
            ) from e
        # n_estimators=8 是默认 ensemble 大小；CPU 上小数据 <1s
        return TabPFNClassifier(n_estimators=8, random_state=seed)

    if backend == "ebm":
        try:
            from interpret.glassbox import ExplainableBoostingClassifier
        except ImportError as e:
            raise ImportError("缺少 interpret 包：pip install interpret") from e
        return ExplainableBoostingClassifier(
            interactions=10,        # 显式交互对数量；n=200 不要太多
            outer_bags=8, inner_bags=0,
            learning_rate=0.02, max_bins=128,
            random_state=seed,
        )

    raise ValueError(f"unknown backend: {backend!r}; "
                     f"choose from {SUPPORTED_BACKENDS}")


def _wrap_isotonic(estimator, cv: int = 3):
    """对树模型套 isotonic 校准（lgbm_mono / xgb 用）。"""
    from sklearn.calibration import CalibratedClassifierCV
    return CalibratedClassifierCV(estimator, method="isotonic", cv=cv)


# ============================================================
#  公开 API：CV 训练 + 一键对比
# ============================================================

def fit_calibrator_cv(X: np.ndarray, y: np.ndarray, backend: str,
                      feature_names: Optional[List[str]] = None,
                      n_splits: int = 5, seed: int = 42,
                      isotonic: Optional[bool] = None) -> Tuple[Calibrator, FoldReport]:
    """K-fold CV 拟合 + OOF 概率上选阈值，最后用全量数据 refit 一份保存。

    isotonic=None 时：tree 类（xgb / lgbm_mono）默认 True，其余 False。
    返回 (在全量上 refit 的 Calibrator, OOF FoldReport)。
    """
    from sklearn.model_selection import StratifiedKFold

    feature_names = feature_names or FEATURE_NAMES_WITH_CLS
    if isotonic is None:
        isotonic = backend in ("xgb", "lgbm_mono")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), -1.0, dtype=np.float64)
    per_fold_f1: List[float] = []

    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        n_pos = max(int(y[tr].sum()), 1)
        n_neg = max(len(tr) - n_pos, 1)
        est = _build_estimator(backend, feature_names, seed=seed + k)
        if backend == "xgb":
            est.set_params(scale_pos_weight=n_neg / n_pos)
        if isotonic:
            # CalibratedClassifierCV 内部还会再切 cv 折，小数据 cv=3 比较稳
            est = _wrap_isotonic(est, cv=min(3, max(2, n_pos // 5)))
        est.fit(X[tr], y[tr])
        p_va = est.predict_proba(X[va])[:, 1]
        oof[va] = p_va
        f_t, f_f1 = find_best_threshold(p_va, y[va])
        per_fold_f1.append(f_f1)
        print(f"  [cv {backend}] fold {k}/{n_splits}  f1={f_f1:.4f} (t={f_t:.2f})")

    best_t, oof_f1 = find_best_threshold(oof, y)
    brier, ll, auc = _brier_logloss_auc(oof, y)
    report = FoldReport(
        backend=backend, oof_probs=oof, labels=y.copy(),
        best_threshold=best_t, f1_oof=oof_f1,
        auc_oof=auc, brier_oof=brier, logloss_oof=ll,
        per_fold_f1=per_fold_f1,
    )

    # 全量 refit 用于推理
    final = _build_estimator(backend, feature_names, seed=seed)
    if backend == "xgb":
        n_pos = max(int(y.sum()), 1)
        n_neg = max(len(y) - n_pos, 1)
        final.set_params(scale_pos_weight=n_neg / n_pos)
    if isotonic:
        final = _wrap_isotonic(final, cv=3)
    final.fit(X, y)
    cal = Calibrator(model=final, threshold=best_t,
                     feature_names=feature_names, backend=backend)
    return cal, report


def compare_backends(X: np.ndarray, y: np.ndarray,
                     backends: List[str],
                     feature_names: Optional[List[str]] = None,
                     n_splits: int = 5, seed: int = 42
                     ) -> Tuple[Dict[str, FoldReport], str]:
    """跑多 backend 5-fold CV，返回 (报告字典, markdown 表格)。"""
    reports: Dict[str, FoldReport] = {}
    for b in backends:
        print(f"\n{'='*60}\n[compare] backend = {b}\n{'='*60}")
        try:
            _, rep = fit_calibrator_cv(X, y, b, feature_names=feature_names,
                                       n_splits=n_splits, seed=seed)
            reports[b] = rep
        except ImportError as e:
            print(f"  [skip] {b}: {e}")

    # 排表
    rows = []
    for b, r in sorted(reports.items(),
                       key=lambda kv: -kv[1].f1_oof):
        rows.append((b, r.f1_oof, r.auc_oof, r.brier_oof,
                     r.logloss_oof, r.best_threshold,
                     float(np.mean(r.per_fold_f1)),
                     float(np.std(r.per_fold_f1))))

    md = ["| backend | OOF F1 | AUC | Brier↓ | LogLoss↓ | best_t | per-fold F1 (mean±std) |",
          "|---|---:|---:|---:|---:|---:|---|"]
    for b, f1, auc, brier, ll, t, m, s in rows:
        md.append(f"| `{b}` | **{f1:.4f}** | {auc:.4f} | {brier:.4f} | "
                  f"{ll:.4f} | {t:.2f} | {m:.4f} ± {s:.4f} |")
    return reports, "\n".join(md)


# ============================================================
#  向后兼容
# ============================================================

def fit_calibrator(X: np.ndarray, y: np.ndarray, backend: str = "logistic",
                   feature_names: Optional[List[str]] = None,
                   seed: int = 42) -> Calibrator:
    """⚠️ DEPRECATED: 旧接口，无 CV 阈值在训练集上偏估。改用 fit_calibrator_cv。"""
    cal, _rep = fit_calibrator_cv(X, y, backend=backend,
                                  feature_names=feature_names,
                                  n_splits=5, seed=seed)
    return cal


def hard_rule_baseline(seg_label: int, cls_score: float,
                       low: float = 0.2, high: float = 0.9) -> int:
    """旧版硬规则 (仅供对照)。"""
    if cls_score < low and seg_label == 1:
        return 0
    if cls_score > high and seg_label == 0:
        return 1
    return seg_label
