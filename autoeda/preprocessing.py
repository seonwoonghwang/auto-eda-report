"""자동 전처리 모듈.

두 단계로 구성된다.

1. **프레임 정제(frame cleaning)** — 제외 컬럼 삭제, 문자열 숫자 복원,
   날짜 컬럼의 파생 변수화, 이상치 윈저라이징 등 DataFrame 수준의 정리.
2. **학습 파이프라인(ColumnTransformer)** — 결측 대체 → 인코딩 → 스케일링을
   scikit-learn 파이프라인으로 구성하여 학습/추론 시 동일하게 재현.

모든 처리 내역은 `PreprocessingReport` 에 기록되어 Word 보고서에 그대로 실린다.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from .config import AnalysisConfig
from .profiling import CATEGORICAL, DATETIME, NUMERIC, TEXT, DataProfile, count_outliers_iqr

#: One-hot 대신 순서형 인코딩으로 전환하는 고유값 임계치
ONEHOT_MAX_CATEGORIES = 30


@dataclass
class PreprocessingReport:
    """전처리 수행 내역."""

    dropped_columns: List[Tuple[str, str]] = field(default_factory=list)
    converted_numeric: List[str] = field(default_factory=list)
    datetime_expanded: List[str] = field(default_factory=list)
    outlier_clipped: List[Tuple[str, int]] = field(default_factory=list)
    missing_filled: List[Tuple[str, str, int]] = field(default_factory=list)
    encoded_onehot: List[str] = field(default_factory=list)
    encoded_ordinal: List[str] = field(default_factory=list)
    scaled_columns: List[str] = field(default_factory=list)
    dropped_rows: int = 0
    notes: List[str] = field(default_factory=list)

    def to_table(self) -> pd.DataFrame:
        """보고서 삽입용 처리 내역 표."""
        rows: List[Dict[str, str]] = []

        for col, reason in self.dropped_columns:
            rows.append({"처리 단계": "컬럼 제외", "대상": col, "내용": reason})
        for col in self.converted_numeric:
            rows.append({"처리 단계": "타입 변환", "대상": col, "내용": "문자열 → 수치형 변환"})
        for col in self.datetime_expanded:
            rows.append(
                {"처리 단계": "날짜 파생", "대상": col, "내용": "연/월/일/요일 파생변수 생성"}
            )
        for col, n in self.outlier_clipped:
            rows.append(
                {"처리 단계": "이상치 처리", "대상": col, "내용": f"IQR 경계로 {n:,}건 조정(윈저라이징)"}
            )
        for col, how, n in self.missing_filled:
            rows.append({"처리 단계": "결측 대체", "대상": col, "내용": f"{how} 대체 {n:,}건"})
        if self.encoded_onehot:
            rows.append(
                {
                    "처리 단계": "인코딩",
                    "대상": f"{len(self.encoded_onehot)}개 컬럼",
                    "내용": "One-hot 인코딩: " + ", ".join(self.encoded_onehot[:10])
                    + ("..." if len(self.encoded_onehot) > 10 else ""),
                }
            )
        if self.encoded_ordinal:
            rows.append(
                {
                    "처리 단계": "인코딩",
                    "대상": f"{len(self.encoded_ordinal)}개 컬럼",
                    "내용": "고유값 과다로 순서형 인코딩: " + ", ".join(self.encoded_ordinal[:10]),
                }
            )
        if self.scaled_columns:
            rows.append(
                {
                    "처리 단계": "스케일링",
                    "대상": f"{len(self.scaled_columns)}개 수치형 컬럼",
                    "내용": "표준화/정규화 적용",
                }
            )
        if self.dropped_rows:
            rows.append(
                {"처리 단계": "행 제거", "대상": "목표 변수 결측", "내용": f"{self.dropped_rows:,}행 제거"}
            )
        for note in self.notes:
            rows.append({"처리 단계": "비고", "대상": "-", "내용": note})

        if not rows:
            rows.append({"처리 단계": "-", "대상": "-", "내용": "별도 전처리가 필요하지 않았습니다."})
        return pd.DataFrame(rows)


@dataclass
class PreparedData:
    """모델링 직전 상태의 데이터 묶음."""

    X: pd.DataFrame
    y: Optional[pd.Series]
    numeric_features: List[str]
    categorical_features: List[str]
    report: PreprocessingReport
    class_labels: Optional[List[str]] = None
    label_mapping: Optional[Dict[str, int]] = None


# ---------------------------------------------------------------------------
# 1단계: DataFrame 수준 정제
# ---------------------------------------------------------------------------
def clean_frame(
    df: pd.DataFrame,
    profile: DataProfile,
    config: AnalysisConfig,
) -> Tuple[pd.DataFrame, PreprocessingReport, List[str], List[str]]:
    """제외/변환/파생/이상치 처리를 수행하고 최종 특성 목록을 돌려준다."""
    report = PreprocessingReport()
    work = df.copy()
    target = config.target

    # (1) 사용자 지정 제외 + 자동 제외 권고
    to_drop: Dict[str, str] = {}
    for col in config.exclude_columns:
        if col in work.columns and col != target:
            to_drop[col] = "사용자 지정 제외"

    for name, cp in profile.columns.items():
        if name == target or name in to_drop or name not in work.columns:
            continue
        if cp.suggest_exclude:
            to_drop[name] = f"자동 제외({cp.exclusion_reason})"

    if to_drop:
        work = work.drop(columns=list(to_drop))
        report.dropped_columns = sorted(to_drop.items())

    # (2) 목표 변수 결측 행 제거
    if target and target in work.columns:
        before = len(work)
        work = work[work[target].notna()]
        report.dropped_rows = before - len(work)

    # (3) 컬럼별 타입 정규화
    numeric_features: List[str] = []
    categorical_features: List[str] = []

    for name in list(work.columns):
        if name == target:
            continue
        kind = config.column_types.get(name) or profile.columns[name].inferred_type

        if kind == NUMERIC:
            if not pd.api.types.is_numeric_dtype(work[name]):
                work[name] = pd.to_numeric(
                    work[name].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                )
                report.converted_numeric.append(name)
            numeric_features.append(name)

        elif kind == DATETIME:
            derived = _expand_datetime(work, name)
            if derived:
                report.datetime_expanded.append(name)
                numeric_features.extend(derived)
            work = work.drop(columns=[name])

        elif kind == TEXT:
            work = work.drop(columns=[name])
            report.dropped_columns.append((name, "텍스트/식별자 컬럼 제외"))

        else:  # CATEGORICAL
            work[name] = work[name].astype("object")
            categorical_features.append(name)

    # (4) 이상치 윈저라이징
    if config.handle_outliers and config.outlier_method == "iqr":
        for name in numeric_features:
            n_out, low, high = count_outliers_iqr(work[name])
            if n_out > 0 and np.isfinite(low) and np.isfinite(high):
                work[name] = work[name].clip(lower=low, upper=high)
                report.outlier_clipped.append((name, n_out))

    # (5) 결측 대체 예정 내역 기록(실제 대체는 파이프라인에서 수행)
    for name in numeric_features + categorical_features:
        n_missing = int(work[name].isna().sum())
        if n_missing:
            how = (
                {"mean": "평균", "median": "중앙값", "constant": "상수(0)"}.get(
                    config.numeric_impute, config.numeric_impute
                )
                if name in numeric_features
                else {"most_frequent": "최빈값", "constant": "상수('미상')"}.get(
                    config.categorical_impute, config.categorical_impute
                )
            )
            report.missing_filled.append((name, how, n_missing))

    if not numeric_features and not categorical_features:
        report.notes.append("사용 가능한 설명 변수가 없습니다. 제외 설정을 확인하세요.")

    return work.reset_index(drop=True), report, numeric_features, categorical_features


def _expand_datetime(df: pd.DataFrame, name: str) -> List[str]:
    """날짜 컬럼을 연/월/일/요일 파생 수치형 컬럼으로 확장한다."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(df[name], errors="coerce", format="mixed")
        except (ValueError, TypeError):
            return []

    if parsed.notna().mean() < 0.5:
        return []

    created: List[str] = []
    for suffix, values in (
        ("연", parsed.dt.year),
        ("월", parsed.dt.month),
        ("일", parsed.dt.day),
        ("요일", parsed.dt.dayofweek),
    ):
        col = f"{name}_{suffix}"
        df[col] = values.astype("float64")
        created.append(col)
    return created


# ---------------------------------------------------------------------------
# 2단계: 학습용 ColumnTransformer
# ---------------------------------------------------------------------------
def build_transformer(
    X: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    config: AnalysisConfig,
    report: Optional[PreprocessingReport] = None,
    *,
    scale: bool = True,
) -> ColumnTransformer:
    """결측 대체 + 인코딩 + 스케일링 ColumnTransformer 를 구성한다.

    scale=False 이면 트리 계열 모델용으로 스케일링을 건너뛴다.
    """
    transformers = []

    if numeric_features:
        steps = [
            (
                "impute",
                SimpleImputer(
                    strategy=config.numeric_impute,
                    fill_value=0 if config.numeric_impute == "constant" else None,
                ),
            )
        ]
        if scale and config.scaling != "none":
            steps.append(
                ("scale", StandardScaler() if config.scaling == "standard" else MinMaxScaler())
            )
            if report is not None and not report.scaled_columns:
                report.scaled_columns = list(numeric_features)
        transformers.append(("num", Pipeline(steps), numeric_features))

    if categorical_features:
        low_card, high_card = [], []
        for col in categorical_features:
            (low_card if X[col].nunique(dropna=True) <= ONEHOT_MAX_CATEGORIES else high_card).append(col)

        if low_card:
            transformers.append(
                (
                    "cat_ohe",
                    Pipeline(
                        [
                            (
                                "impute",
                                SimpleImputer(
                                    strategy=config.categorical_impute,
                                    fill_value="미상"
                                    if config.categorical_impute == "constant"
                                    else None,
                                ),
                            ),
                            (
                                "encode",
                                OneHotEncoder(
                                    handle_unknown="infrequent_if_exist",
                                    sparse_output=False,
                                    min_frequency=0.01,
                                ),
                            ),
                        ]
                    ),
                    low_card,
                )
            )
            if report is not None and not report.encoded_onehot:
                report.encoded_onehot = list(low_card)

        if high_card:
            transformers.append(
                (
                    "cat_ord",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            (
                                "encode",
                                OrdinalEncoder(
                                    handle_unknown="use_encoded_value", unknown_value=-1
                                ),
                            ),
                        ]
                    ),
                    high_card,
                )
            )
            if report is not None and not report.encoded_ordinal:
                report.encoded_ordinal = list(high_card)

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


# ---------------------------------------------------------------------------
# 목표 변수 준비
# ---------------------------------------------------------------------------
def prepare_target(
    series: pd.Series, task_type: str
) -> Tuple[pd.Series, Optional[List[str]], Optional[Dict[str, int]]]:
    """목표 변수를 모델 학습용으로 변환한다.

    분류: 문자열 라벨 → 정수 코드, 원본 라벨 목록과 매핑을 함께 반환.
    회귀: 수치형으로 강제 변환.
    """
    if task_type == "classification":
        labels = series.astype(str)
        categories = sorted(labels.unique())
        mapping = {label: idx for idx, label in enumerate(categories)}
        return labels.map(mapping).astype(int), categories, mapping

    numeric = pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False), errors="coerce"
    ) if not pd.api.types.is_numeric_dtype(series) else series.astype("float64")
    return numeric, None, None


def prepare_data(
    df: pd.DataFrame,
    profile: DataProfile,
    config: AnalysisConfig,
    task_type: str,
) -> PreparedData:
    """정제 → 특성/타깃 분리까지 수행한 결과를 반환한다."""
    cleaned, report, numeric_features, categorical_features = clean_frame(df, profile, config)

    y: Optional[pd.Series] = None
    labels = mapping = None
    if config.target and config.target in cleaned.columns and task_type != "eda":
        y, labels, mapping = prepare_target(cleaned[config.target], task_type)
        valid = y.notna()
        if not valid.all():
            report.dropped_rows += int((~valid).sum())
            cleaned = cleaned[valid].reset_index(drop=True)
            y = y[valid].reset_index(drop=True)

    feature_cols = [c for c in numeric_features + categorical_features if c in cleaned.columns]
    X = cleaned[feature_cols].copy()

    return PreparedData(
        X=X,
        y=y,
        numeric_features=[c for c in numeric_features if c in X.columns],
        categorical_features=[c for c in categorical_features if c in X.columns],
        report=report,
        class_labels=labels,
        label_mapping=mapping,
    )
