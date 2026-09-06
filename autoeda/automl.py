"""AutoML 엔진.

여러 알고리즘을 동일 조건(교차검증)으로 비교하여 최적 모델을 자동 선택하고,
홀드아웃 성능 지표와 변수 중요도를 산출한다.

XGBoost / LightGBM 은 설치되어 있으면 자동으로 후보에 포함되고,
없으면 scikit-learn 내장 부스팅 모델로 대체된다(설치 강제 없음).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .config import AnalysisConfig
from .preprocessing import PreparedData, build_transformer

# 선택적 의존성 ---------------------------------------------------------------
try:  # pragma: no cover - 설치 환경에 따라 달라짐
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGBOOST = True
except Exception:  # noqa: BLE001
    HAS_XGBOOST = False

try:  # pragma: no cover
    from lightgbm import LGBMClassifier, LGBMRegressor

    HAS_LIGHTGBM = True
except Exception:  # noqa: BLE001
    HAS_LIGHTGBM = False


@dataclass
class ModelScore:
    """개별 후보 모델의 교차검증 결과."""

    name: str
    metrics: Dict[str, float]
    fit_seconds: float
    failed: bool = False
    error: str = ""


@dataclass
class ModelingResult:
    """모델링 전체 결과."""

    task_type: str
    target: str
    best_model_name: str
    best_pipeline: Any
    leaderboard: pd.DataFrame
    holdout_metrics: Dict[str, float]
    feature_importance: pd.DataFrame
    n_train: int
    n_test: int
    n_features_out: int
    class_labels: Optional[List[str]] = None
    y_test: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    y_proba: Optional[np.ndarray] = None
    confusion: Optional[np.ndarray] = None
    roc: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
    warnings_: List[str] = field(default_factory=list)

    def leaderboard_display(self) -> pd.DataFrame:
        """보고서용 리더보드(한국어 컬럼, 반올림)."""
        if self.leaderboard.empty:
            return self.leaderboard
        df = self.leaderboard.copy()
        num_cols = [c for c in df.columns if c != "모델"]
        df[num_cols] = df[num_cols].astype(float).round(4)
        return df

    def holdout_table(self) -> pd.DataFrame:
        rows = [{"지표": k, "값": round(float(v), 4)} for k, v in self.holdout_metrics.items()]
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 후보 모델 정의
# ---------------------------------------------------------------------------
#: (표시명, 추정기 팩토리, 스케일링 필요 여부)
def regression_candidates(seed: int) -> List[Tuple[str, Any, bool]]:
    cands: List[Tuple[str, Any, bool]] = [
        ("선형 회귀", LinearRegression(), True),
        ("릿지 회귀", Ridge(alpha=1.0, random_state=seed), True),
        ("엘라스틱넷", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=seed, max_iter=5000), True),
        ("의사결정나무", DecisionTreeRegressor(max_depth=8, random_state=seed), False),
        ("랜덤 포레스트", RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1), False),
        ("엑스트라 트리", ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=-1), False),
        ("그래디언트 부스팅", GradientBoostingRegressor(random_state=seed), False),
        ("히스토그램 부스팅", HistGradientBoostingRegressor(random_state=seed), False),
        ("K-최근접 이웃", KNeighborsRegressor(n_neighbors=5), True),
    ]
    if HAS_XGBOOST:
        cands.append(
            ("XGBoost", XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=6,
                                     subsample=0.9, colsample_bytree=0.9,
                                     random_state=seed, n_jobs=-1, verbosity=0), False)
        )
    if HAS_LIGHTGBM:
        cands.append(
            ("LightGBM", LGBMRegressor(n_estimators=400, learning_rate=0.08,
                                       random_state=seed, n_jobs=-1, verbose=-1), False)
        )
    return cands


def classification_candidates(seed: int) -> List[Tuple[str, Any, bool]]:
    cands: List[Tuple[str, Any, bool]] = [
        ("로지스틱 회귀", LogisticRegression(max_iter=2000, random_state=seed), True),
        ("의사결정나무", DecisionTreeClassifier(max_depth=8, random_state=seed), False),
        ("랜덤 포레스트", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1), False),
        ("엑스트라 트리", ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=-1), False),
        ("그래디언트 부스팅", GradientBoostingClassifier(random_state=seed), False),
        ("히스토그램 부스팅", HistGradientBoostingClassifier(random_state=seed), False),
        ("K-최근접 이웃", KNeighborsClassifier(n_neighbors=5), True),
    ]
    if HAS_XGBOOST:
        cands.append(
            ("XGBoost", XGBClassifier(n_estimators=400, learning_rate=0.08, max_depth=6,
                                      subsample=0.9, colsample_bytree=0.9, random_state=seed,
                                      n_jobs=-1, verbosity=0, eval_metric="logloss"), False)
        )
    if HAS_LIGHTGBM:
        cands.append(
            ("LightGBM", LGBMClassifier(n_estimators=400, learning_rate=0.08,
                                        random_state=seed, n_jobs=-1, verbose=-1), False)
        )
    return cands


REGRESSION_SCORING = {
    "R2": "r2",
    "RMSE": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
}
CLASSIFICATION_SCORING = {
    "정확도": "accuracy",
    "F1(가중)": "f1_weighted",
    "정밀도(가중)": "precision_weighted",
    "재현율(가중)": "recall_weighted",
}
PRIMARY_METRIC = {"regression": "R2", "classification": "F1(가중)"}


# ---------------------------------------------------------------------------
# 메인 실행
# ---------------------------------------------------------------------------
def run_automl(
    data: PreparedData,
    config: AnalysisConfig,
    task_type: str,
    progress=None,
) -> ModelingResult:
    """후보 모델 비교 → 최적 모델 선정 → 홀드아웃 평가 → 변수 중요도 산출."""
    if data.y is None:
        raise ValueError("목표 변수가 지정되지 않아 모델링을 수행할 수 없습니다.")
    if data.X.empty or data.X.shape[1] == 0:
        raise ValueError("사용 가능한 설명 변수가 없습니다. 컬럼 제외 설정을 확인하세요.")

    notes: List[str] = []
    seed = config.random_state
    X, y = data.X, data.y.reset_index(drop=True)

    # 1) 학습/검증 분할 ---------------------------------------------------
    stratify = None
    if task_type == "classification":
        counts = y.value_counts()
        if counts.min() >= 2 and len(counts) < len(y) * 0.5:
            stratify = y
        else:
            notes.append("일부 클래스의 표본이 부족하여 층화 분할을 적용하지 못했습니다.")

    test_size = config.test_size
    if len(X) < 30:
        test_size = max(0.2, min(0.3, test_size))
        notes.append(f"표본이 {len(X)}건으로 적어 성능 지표의 신뢰구간이 넓을 수 있습니다.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    # 2) 교차검증 설정 -----------------------------------------------------
    n_splits = int(config.cv_folds)
    if task_type == "classification":
        min_class = int(y_train.value_counts().min())
        n_splits = max(2, min(n_splits, min_class))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        n_splits = max(2, min(n_splits, len(X_train)))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    scoring = REGRESSION_SCORING if task_type == "regression" else CLASSIFICATION_SCORING
    primary = PRIMARY_METRIC[task_type]

    candidates = (
        regression_candidates(seed) if task_type == "regression" else classification_candidates(seed)
    )[: config.max_models]

    # 3) 후보 모델 비교 ----------------------------------------------------
    scores: List[ModelScore] = []
    pipelines: Dict[str, Pipeline] = {}

    for idx, (name, estimator, needs_scale) in enumerate(candidates, start=1):
        if progress:
            progress(idx / (len(candidates) + 1), f"모델 비교 중… ({idx}/{len(candidates)}) {name}")

        transformer = build_transformer(
            X_train,
            data.numeric_features,
            data.categorical_features,
            config,
            data.report,
            scale=needs_scale,
        )
        pipe = Pipeline([("prep", transformer), ("model", estimator)])
        pipelines[name] = pipe

        started = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_out = cross_validate(
                    pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1, error_score="raise"
                )
            metrics = {
                label: _normalize_score(label, float(np.mean(cv_out[f"test_{label}"])))
                for label in scoring
            }
            scores.append(ModelScore(name=name, metrics=metrics,
                                     fit_seconds=time.perf_counter() - started))
        except Exception as exc:  # noqa: BLE001 - 후보 실패는 건너뛴다
            scores.append(
                ModelScore(name=name, metrics={}, fit_seconds=time.perf_counter() - started,
                           failed=True, error=str(exc)[:200])
            )

    valid_scores = [s for s in scores if not s.failed]
    if not valid_scores:
        failed_msgs = "; ".join(f"{s.name}: {s.error}" for s in scores[:3])
        raise RuntimeError(f"모든 후보 모델의 학습에 실패했습니다. ({failed_msgs})")

    for s in scores:
        if s.failed:
            notes.append(f"{s.name} 모델은 학습에 실패하여 제외되었습니다.")

    # 4) 최적 모델 선정 ----------------------------------------------------
    higher_is_better = True  # R2, F1 모두 클수록 좋음
    best = max(valid_scores, key=lambda s: s.metrics.get(primary, -np.inf))
    if not higher_is_better:  # pragma: no cover - 대칭성 유지용
        best = min(valid_scores, key=lambda s: s.metrics.get(primary, np.inf))

    leaderboard = pd.DataFrame(
        [{"모델": s.name, **s.metrics, "학습시간(초)": round(s.fit_seconds, 2)} for s in valid_scores]
    ).sort_values(primary, ascending=False).reset_index(drop=True)
    leaderboard.insert(0, "순위", range(1, len(leaderboard) + 1))

    # 5) 최적 모델 최종 학습 및 홀드아웃 평가 --------------------------------
    if progress:
        progress(0.9, f"최적 모델 학습 중… ({best.name})")

    best_pipe = pipelines[best.name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_pipe.fit(X_train, y_train)
        y_pred = best_pipe.predict(X_test)

    y_proba = None
    confusion = None
    roc_data = None

    if task_type == "regression":
        holdout = _regression_metrics(y_test, y_pred)
        baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
        holdout["기준모델 대비 RMSE 개선율"] = _improvement(
            mean_squared_error(y_test, baseline.predict(X_test)) ** 0.5, holdout["RMSE"]
        )
    else:
        holdout = _classification_metrics(y_test, y_pred)
        labels_idx = sorted(pd.unique(pd.concat([pd.Series(y_test), pd.Series(y_pred)])))
        confusion = confusion_matrix(y_test, y_pred, labels=labels_idx)
        if hasattr(best_pipe, "predict_proba"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y_proba = best_pipe.predict_proba(X_test)
                if y_proba.shape[1] == 2:
                    auc = roc_auc_score(y_test, y_proba[:, 1])
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                    roc_data = (fpr, tpr, float(auc))
                    holdout["ROC-AUC"] = float(auc)
                else:
                    holdout["ROC-AUC(OvR)"] = float(
                        roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
                    )
            except Exception as exc:  # noqa: BLE001
                notes.append(f"확률 기반 지표 계산 생략: {str(exc)[:120]}")

        baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
        holdout["기준모델 대비 정확도 개선율"] = _improvement(
            holdout["정확도"], accuracy_score(y_test, baseline.predict(X_test)), invert=True
        )

    # 6) 변수 중요도 --------------------------------------------------------
    importance = pd.DataFrame(columns=["변수", "중요도", "중요도(%)"])
    if config.compute_importance:
        if progress:
            progress(0.96, "변수 중요도 계산 중…")
        importance = compute_feature_importance(
            best_pipe, X_test, y_test, task_type, seed, notes
        )

    n_features_out = _n_output_features(best_pipe, X_train)

    return ModelingResult(
        task_type=task_type,
        target=config.target or "",
        best_model_name=best.name,
        best_pipeline=best_pipe,
        leaderboard=leaderboard,
        holdout_metrics=holdout,
        feature_importance=importance,
        n_train=len(X_train),
        n_test=len(X_test),
        n_features_out=n_features_out,
        class_labels=data.class_labels,
        y_test=np.asarray(y_test),
        y_pred=np.asarray(y_pred),
        y_proba=y_proba,
        confusion=confusion,
        roc=roc_data,
        warnings_=notes,
    )


# ---------------------------------------------------------------------------
# 보조 함수
# ---------------------------------------------------------------------------
def _normalize_score(label: str, value: float) -> float:
    """neg_* 스코어를 양수 지표로 되돌린다."""
    return abs(value) if label in {"RMSE", "MAE"} else value


def _regression_metrics(y_true, y_pred) -> Dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    y_true_arr = np.asarray(y_true, dtype=float)
    denom = np.where(np.abs(y_true_arr) < 1e-9, np.nan, np.abs(y_true_arr))
    mape = float(np.nanmean(np.abs((y_true_arr - np.asarray(y_pred, dtype=float)) / denom)) * 100)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": mse ** 0.5,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE(%)": mape,
    }


def _classification_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "정확도": float(accuracy_score(y_true, y_pred)),
        "F1(가중)": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "정밀도(가중)": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "재현율(가중)": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _improvement(baseline: float, model: float, invert: bool = False) -> float:
    """기준 모델 대비 개선율(%). invert=True 면 클수록 좋은 지표."""
    try:
        if invert:
            return float((baseline - model) / abs(model) * 100) if model else 0.0
        return float((baseline - model) / abs(baseline) * 100) if baseline else 0.0
    except ZeroDivisionError:  # pragma: no cover
        return 0.0


def _n_output_features(pipe: Pipeline, X: pd.DataFrame) -> int:
    try:
        return int(len(pipe.named_steps["prep"].get_feature_names_out()))
    except Exception:  # noqa: BLE001
        return int(X.shape[1])


def compute_feature_importance(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test,
    task_type: str,
    seed: int,
    notes: List[str],
) -> pd.DataFrame:
    """원본 컬럼 기준 변수 중요도를 계산한다.

    permutation importance 를 원본 컬럼 단위로 적용하므로, One-hot 으로 분해된
    더미 변수가 아니라 실무자가 이해할 수 있는 **원래 컬럼 이름**으로 결과가 나온다.
    """
    n_repeats = 10 if len(X_test) <= 5000 else 5
    scoring = "r2" if task_type == "regression" else "f1_weighted"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = permutation_importance(
                pipe,
                X_test,
                y_test,
                n_repeats=n_repeats,
                random_state=seed,
                scoring=scoring,
                n_jobs=1,
            )
        values = np.clip(result.importances_mean, a_min=0, a_max=None)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"순열 중요도 계산에 실패하여 모델 내장 중요도로 대체합니다. ({str(exc)[:100]})")
        return _fallback_importance(pipe, X_test)

    total = values.sum()
    df = pd.DataFrame(
        {
            "변수": list(X_test.columns),
            "중요도": values,
            "중요도(%)": (values / total * 100) if total > 0 else 0.0,
        }
    )
    df = df.sort_values("중요도", ascending=False).reset_index(drop=True)
    df["중요도"] = df["중요도"].round(6)
    df["중요도(%)"] = df["중요도(%)"].round(2)
    return df


def _fallback_importance(pipe: Pipeline, X_test: pd.DataFrame) -> pd.DataFrame:
    """모델 내장 feature_importances_ / coef_ 를 사용한 대체 중요도."""
    model = pipe.named_steps.get("model")
    try:
        names = list(pipe.named_steps["prep"].get_feature_names_out())
    except Exception:  # noqa: BLE001
        names = list(X_test.columns)

    values: Optional[np.ndarray] = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

    if values is None or len(values) != len(names):
        return pd.DataFrame(columns=["변수", "중요도", "중요도(%)"])

    total = values.sum()
    df = pd.DataFrame(
        {
            "변수": names,
            "중요도": values.round(6),
            "중요도(%)": (values / total * 100).round(2) if total > 0 else 0.0,
        }
    )
    return df.sort_values("중요도", ascending=False).reset_index(drop=True)
