"""데이터 프로파일링 모듈.

컬럼 타입 자동 추론, 결측/이상치 스캔, 기초 통계량과 상관계수 매트릭스 계산을
담당한다. 여기서 만든 프로파일이 전처리·모델링·보고서의 공통 입력이 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import (
    CATEGORICAL_UNIQUE_THRESHOLD,
    CLASSIFICATION_MAX_CLASSES,
    HIGH_CARDINALITY_RATIO,
    MISSING_DROP_RATIO,
)

NUMERIC = "numeric"
CATEGORICAL = "categorical"
DATETIME = "datetime"
TEXT = "text"


@dataclass
class ColumnProfile:
    """개별 컬럼의 프로파일 정보."""

    name: str
    inferred_type: str
    dtype: str
    n_missing: int
    missing_ratio: float
    n_unique: int
    unique_ratio: float
    sample_values: List[str] = field(default_factory=list)
    # 수치형 전용
    mean: Optional[float] = None
    std: Optional[float] = None
    minimum: Optional[float] = None
    q1: Optional[float] = None
    median: Optional[float] = None
    q3: Optional[float] = None
    maximum: Optional[float] = None
    skewness: Optional[float] = None
    n_outliers: int = 0
    outlier_ratio: float = 0.0
    # 범주형 전용
    top_value: Optional[str] = None
    top_freq: Optional[int] = None
    #: 자동 제외 권고 사유 (없으면 빈 문자열)
    exclusion_reason: str = ""

    @property
    def suggest_exclude(self) -> bool:
        return bool(self.exclusion_reason)


@dataclass
class DataProfile:
    """데이터셋 전체 프로파일."""

    n_rows: int
    n_cols: int
    columns: Dict[str, ColumnProfile]
    memory_mb: float
    n_duplicated_rows: int
    correlation: Optional[pd.DataFrame] = None

    # 편의 접근자 ---------------------------------------------------------
    def names_by_type(self, kind: str) -> List[str]:
        return [c.name for c in self.columns.values() if c.inferred_type == kind]

    @property
    def numeric_columns(self) -> List[str]:
        return self.names_by_type(NUMERIC)

    @property
    def categorical_columns(self) -> List[str]:
        return self.names_by_type(CATEGORICAL)

    @property
    def datetime_columns(self) -> List[str]:
        return self.names_by_type(DATETIME)

    @property
    def text_columns(self) -> List[str]:
        return self.names_by_type(TEXT)

    @property
    def suggested_exclusions(self) -> List[str]:
        return [c.name for c in self.columns.values() if c.suggest_exclude]

    # 표 형태 출력 --------------------------------------------------------
    def overview_table(self) -> pd.DataFrame:
        """컬럼별 요약표(보고서 및 GUI 공용)."""
        rows = []
        for c in self.columns.values():
            rows.append(
                {
                    "컬럼명": c.name,
                    "추론 타입": _KOR_TYPE.get(c.inferred_type, c.inferred_type),
                    "원본 dtype": c.dtype,
                    "결측 수": c.n_missing,
                    "결측률(%)": round(c.missing_ratio * 100, 2),
                    "고유값 수": c.n_unique,
                    "대표값/평균": _headline_value(c),
                }
            )
        return pd.DataFrame(rows)

    def numeric_describe_table(self) -> pd.DataFrame:
        """수치형 기술통계표."""
        rows = []
        for name in self.numeric_columns:
            c = self.columns[name]
            rows.append(
                {
                    "컬럼명": name,
                    "평균": c.mean,
                    "표준편차": c.std,
                    "최소": c.minimum,
                    "1사분위": c.q1,
                    "중앙값": c.median,
                    "3사분위": c.q3,
                    "최대": c.maximum,
                    "왜도": c.skewness,
                    "이상치 수": c.n_outliers,
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            numeric_cols = df.columns.drop("컬럼명")
            df[numeric_cols] = df[numeric_cols].astype(float).round(4)
        return df

    def categorical_table(self, top_n: int = 30) -> pd.DataFrame:
        """범주형 요약표."""
        rows = []
        for name in self.categorical_columns[:top_n]:
            c = self.columns[name]
            rows.append(
                {
                    "컬럼명": name,
                    "고유값 수": c.n_unique,
                    "최빈값": c.top_value,
                    "최빈 빈도": c.top_freq,
                    "결측률(%)": round(c.missing_ratio * 100, 2),
                }
            )
        return pd.DataFrame(rows)


_KOR_TYPE = {
    NUMERIC: "수치형",
    CATEGORICAL: "범주형",
    DATETIME: "날짜형",
    TEXT: "텍스트/식별자",
}


def _headline_value(c: ColumnProfile) -> str:
    if c.inferred_type == NUMERIC and c.mean is not None:
        return f"{c.mean:,.4g}"
    if c.top_value is not None:
        return str(c.top_value)
    return ", ".join(c.sample_values[:2])


# ---------------------------------------------------------------------------
# 타입 추론
# ---------------------------------------------------------------------------
def infer_column_type(series: pd.Series) -> str:
    """단일 컬럼의 분석용 타입을 추론한다."""
    s = series.dropna()

    if pd.api.types.is_datetime64_any_dtype(series):
        return DATETIME
    if pd.api.types.is_bool_dtype(series):
        return CATEGORICAL
    if isinstance(series.dtype, pd.CategoricalDtype):
        return CATEGORICAL

    if s.empty:
        return TEXT

    if pd.api.types.is_numeric_dtype(series):
        n_unique = s.nunique()
        # 정수이면서 고유값이 매우 적으면 코드성 범주로 간주
        if n_unique <= CATEGORICAL_UNIQUE_THRESHOLD and _looks_integer(s):
            return CATEGORICAL
        return NUMERIC

    # 문자열: 숫자로 변환 가능한지, 날짜로 변환 가능한지 확인
    sample = s.astype(str).head(2000)

    converted = pd.to_numeric(sample.str.replace(",", "", regex=False), errors="coerce")
    if converted.notna().mean() > 0.95:
        return NUMERIC if converted.nunique() > CATEGORICAL_UNIQUE_THRESHOLD else CATEGORICAL

    if _looks_datetime(sample):
        return DATETIME

    n_unique = s.nunique()
    if n_unique / max(len(s), 1) >= HIGH_CARDINALITY_RATIO and n_unique > CATEGORICAL_UNIQUE_THRESHOLD:
        return TEXT
    return CATEGORICAL


def _looks_integer(s: pd.Series) -> bool:
    try:
        arr = s.to_numpy(dtype="float64", copy=False)
    except (TypeError, ValueError):
        return False
    finite = arr[np.isfinite(arr)]
    return finite.size > 0 and bool(np.all(np.equal(np.mod(finite, 1), 0)))


def _looks_datetime(sample: pd.Series) -> bool:
    """문자열 표본이 날짜 형식인지 판별한다(경고 없이 조용히 시도)."""
    import warnings

    if not sample.str.contains(r"[-/:]|년|월", regex=True, na=False).mean() > 0.6:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            return False
    return bool(parsed.notna().mean() > 0.9)


def count_outliers_iqr(series: pd.Series) -> tuple[int, float, float]:
    """IQR 기준 이상치 개수와 하/상한 경계를 반환한다."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0, float("nan"), float("nan")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0, float(q1), float(q3)
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < low) | (s > high)).sum()), float(low), float(high)


# ---------------------------------------------------------------------------
# 전체 프로파일링
# ---------------------------------------------------------------------------
def profile_dataframe(
    df: pd.DataFrame,
    *,
    overrides: Optional[Dict[str, str]] = None,
    correlation_max_cols: int = 40,
) -> DataProfile:
    """DataFrame 전체를 프로파일링한다.

    overrides 로 사용자가 GUI에서 수정한 컬럼 타입을 우선 적용할 수 있다.
    """
    overrides = overrides or {}
    n_rows = len(df)
    profiles: Dict[str, ColumnProfile] = {}

    for name in df.columns:
        series = df[name]
        inferred = overrides.get(name) or infer_column_type(series)

        n_missing = int(series.isna().sum())
        missing_ratio = n_missing / n_rows if n_rows else 0.0
        n_unique = int(series.nunique(dropna=True))
        unique_ratio = n_unique / n_rows if n_rows else 0.0

        sample_values = [
            str(v) for v in series.dropna().unique()[:5]
        ]

        cp = ColumnProfile(
            name=name,
            inferred_type=inferred,
            dtype=str(series.dtype),
            n_missing=n_missing,
            missing_ratio=missing_ratio,
            n_unique=n_unique,
            unique_ratio=unique_ratio,
            sample_values=sample_values,
        )

        if inferred == NUMERIC:
            s = pd.to_numeric(series, errors="coerce")
            desc = s.describe()
            cp.mean = _f(desc.get("mean"))
            cp.std = _f(desc.get("std"))
            cp.minimum = _f(desc.get("min"))
            cp.q1 = _f(desc.get("25%"))
            cp.median = _f(desc.get("50%"))
            cp.q3 = _f(desc.get("75%"))
            cp.maximum = _f(desc.get("max"))
            try:
                cp.skewness = _f(s.skew())
            except (TypeError, ValueError):
                cp.skewness = None
            n_out, _, _ = count_outliers_iqr(s)
            cp.n_outliers = n_out
            cp.outlier_ratio = n_out / n_rows if n_rows else 0.0
        else:
            vc = series.value_counts(dropna=True)
            if not vc.empty:
                cp.top_value = str(vc.index[0])
                cp.top_freq = int(vc.iloc[0])

        # 자동 제외 권고 사유 판정
        if missing_ratio > MISSING_DROP_RATIO:
            cp.exclusion_reason = f"결측률 {missing_ratio:.0%}로 과다"
        elif n_unique <= 1:
            cp.exclusion_reason = "값이 한 종류뿐(정보량 없음)"
        elif inferred == TEXT:
            cp.exclusion_reason = "식별자/자유 텍스트로 추정"

        profiles[name] = cp

    # 상관계수 매트릭스 -----------------------------------------------------
    numeric_names = [n for n, p in profiles.items() if p.inferred_type == NUMERIC]
    correlation = None
    if len(numeric_names) >= 2:
        subset = numeric_names[:correlation_max_cols]
        correlation = df[subset].apply(pd.to_numeric, errors="coerce").corr()

    return DataProfile(
        n_rows=n_rows,
        n_cols=df.shape[1],
        columns=profiles,
        memory_mb=float(df.memory_usage(deep=True).sum()) / 1024 / 1024,
        n_duplicated_rows=int(df.duplicated().sum()),
        correlation=correlation,
    )


def _f(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(out) else out


# ---------------------------------------------------------------------------
# 문제 유형 판정
# ---------------------------------------------------------------------------
def suggest_task_type(df: pd.DataFrame, target: Optional[str]) -> str:
    """목표 변수로부터 회귀/분류 여부를 추정한다."""
    if not target or target not in df.columns:
        return "eda"

    s = df[target].dropna()
    if s.empty:
        return "eda"

    if pd.api.types.is_bool_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return "classification"

    if pd.api.types.is_numeric_dtype(s):
        n_unique = s.nunique()
        if n_unique <= 2:
            return "classification"
        if n_unique <= CLASSIFICATION_MAX_CLASSES and _looks_integer(s):
            return "classification"
        return "regression"

    numeric_try = pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")
    if numeric_try.notna().mean() > 0.95 and numeric_try.nunique() > CLASSIFICATION_MAX_CLASSES:
        return "regression"
    return "classification"


def top_correlations(corr: Optional[pd.DataFrame], top_n: int = 15) -> pd.DataFrame:
    """상관계수 매트릭스에서 절댓값이 큰 변수쌍을 뽑아 표로 반환한다."""
    if corr is None or corr.empty:
        return pd.DataFrame(columns=["변수 1", "변수 2", "상관계수"])

    pairs = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            value = corr.loc[a, b]
            if pd.notna(value):
                pairs.append({"변수 1": a, "변수 2": b, "상관계수": round(float(value), 4)})

    out = pd.DataFrame(pairs)
    if out.empty:
        return out
    out["_abs"] = out["상관계수"].abs()
    return out.sort_values("_abs", ascending=False).drop(columns="_abs").head(top_n).reset_index(drop=True)
