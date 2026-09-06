"""차트 생성 모듈.

모든 차트는 PNG 파일로 저장되어 Word 보고서에 삽입된다. 정적 이미지이므로
호버/툴팁 대신 **선택적 직접 라벨**과 표로 값을 읽을 수 있게 한다.

색상 규칙
---------
* 계열 구분(범주형) : `config.PALETTE` 를 고정 순서로 사용, 순환하지 않음
* 크기(연속형)     : 파랑 단일 색상 램프
* 극성(상관계수)   : 파랑 ↔ 회색 ↔ 빨강 발산형 (중앙은 무채색)
* 축·격자는 배경으로 물러나고, 텍스트는 잉크 색을 사용
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from .config import (  # noqa: E402
    AXIS_COLOR,
    CHART_FIGSIZE,
    DIVERGING_HIGH,
    DIVERGING_LOW,
    DIVERGING_MID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    PALETTE,
    SEQUENTIAL_RAMP,
    apply_matplotlib_style,
)
from .profiling import DataProfile

SEQ_CMAP = LinearSegmentedColormap.from_list("autoeda_seq", SEQUENTIAL_RAMP)
DIV_CMAP = LinearSegmentedColormap.from_list(
    "autoeda_div", [DIVERGING_LOW, DIVERGING_MID, DIVERGING_HIGH]
)


@dataclass
class Chart:
    """생성된 차트 한 장."""

    key: str
    title: str
    path: Path
    caption: str = ""


@dataclass
class ChartSet:
    """보고서에 삽입할 차트 모음."""

    charts: List[Chart] = field(default_factory=list)
    font_name: str = ""
    font_warning: str = ""

    def add(self, chart: Optional[Chart]) -> None:
        if chart is not None:
            self.charts.append(chart)

    def by_key(self, key: str) -> List[Chart]:
        return [c for c in self.charts if c.key == key]

    def __len__(self) -> int:  # pragma: no cover - 편의용
        return len(self.charts)


# ---------------------------------------------------------------------------
# 공통 헬퍼
# ---------------------------------------------------------------------------
def init_style() -> tuple[str, str]:
    """스타일을 적용하고 (폰트명, 경고문) 을 반환한다."""
    font = apply_matplotlib_style()
    plt.rcParams["axes.axisbelow"] = True
    warning = (
        ""
        if font != "DejaVu Sans"
        else "한글 폰트를 찾지 못해 차트의 한글이 깨질 수 있습니다. "
        "'맑은 고딕' 또는 '나눔고딕' 설치를 권장합니다."
    )
    return font, warning


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    with warnings.catch_warnings():
        # 한글 폰트가 없는 환경의 글리프 경고는 별도 안내 문구로 대체한다.
        warnings.filterwarnings("ignore", message=".*missing from font.*")
        fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _shorten(label: str, limit: int = 22) -> str:
    text = str(label)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _pad_bar_axis(ax, n_items: int, minimum: int = 5) -> None:
    """항목이 적을 때 막대가 과도하게 두꺼워지지 않도록 범주 축 여백을 확보한다."""
    if n_items < minimum:
        ax.set_ylim(-0.6, minimum - 0.4)


def _style_value_axis(ax, axis: str = "y") -> None:
    """값 축에만 격자를 남기고 범주 축 격자는 제거한다."""
    ax.grid(axis=axis, visible=True)
    ax.grid(axis="x" if axis == "y" else "y", visible=False)


# ---------------------------------------------------------------------------
# 1. 탐색적 분석 차트
# ---------------------------------------------------------------------------
def plot_missing_values(profile: DataProfile, out_dir: Path, top_n: int = 20) -> Optional[Chart]:
    """결측률 상위 컬럼 가로 막대."""
    items = [
        (c.name, c.missing_ratio * 100)
        for c in profile.columns.values()
        if c.n_missing > 0
    ]
    if not items:
        return None

    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:top_n][::-1]
    names = [_shorten(n) for n, _ in items]
    values = [v for _, v in items]

    height = max(2.6, 0.34 * len(items) + 1.2)
    fig, ax = plt.subplots(figsize=(CHART_FIGSIZE[0], height))
    bars = ax.barh(names, values, color=PALETTE[0], height=0.62)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=8.5,
            color=INK_SECONDARY,
        )
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xlabel("결측률 (%)")
    ax.set_title("컬럼별 결측률")
    _pad_bar_axis(ax, len(items))
    _style_value_axis(ax, axis="x")
    return Chart(
        key="missing",
        title="컬럼별 결측률",
        path=_save(fig, out_dir, "missing_values"),
        caption="결측률이 높은 컬럼은 대체 결과의 신뢰도가 낮으므로 해석에 주의가 필요합니다.",
    )


def plot_numeric_distributions(
    df: pd.DataFrame, profile: DataProfile, out_dir: Path, max_cols: int = 6
) -> List[Chart]:
    """주요 수치형 변수의 분포(히스토그램 + 중앙값 표시)."""
    charts: List[Chart] = []
    targets = profile.numeric_columns[:max_cols]
    if not targets:
        return charts

    n = len(targets)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(CHART_FIGSIZE[0], 2.5 * nrows + 0.6), squeeze=False
    )

    for idx, name in enumerate(targets):
        ax = axes[idx // ncols][idx % ncols]
        values = pd.to_numeric(df[name], errors="coerce").dropna()
        if values.empty:
            ax.axis("off")
            continue
        bins = min(40, max(10, int(np.sqrt(len(values)))))
        ax.hist(values, bins=bins, color=PALETTE[0], edgecolor="white", linewidth=0.6)
        median = float(values.median())
        ax.axvline(median, color=INK_SECONDARY, linestyle="--", linewidth=1.2)
        ax.text(
            0.98,
            0.94,
            f"중앙값 {median:,.4g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color=INK_SECONDARY,
        )
        ax.set_title(_shorten(name, 26), fontsize=10.5)
        ax.set_ylabel("빈도")
        _style_value_axis(ax, axis="y")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("주요 수치형 변수 분포", fontsize=12, fontweight="bold", color=INK_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    charts.append(
        Chart(
            key="distribution",
            title="주요 수치형 변수 분포",
            path=_save(fig, out_dir, "distributions"),
            caption="점선은 각 변수의 중앙값입니다. 분포가 한쪽으로 치우친 변수는 로그 변환을 검토할 수 있습니다.",
        )
    )
    return charts


def plot_correlation_heatmap(
    corr: Optional[pd.DataFrame], out_dir: Path, max_cols: int = 15
) -> Optional[Chart]:
    """상관계수 히트맵(발산형 색상, 중앙 0 = 무채색)."""
    if corr is None or corr.shape[0] < 2:
        return None

    if corr.shape[0] > max_cols:
        # 다른 변수와의 평균 상관 강도가 큰 상위 변수만 표시
        strength = corr.abs().mean().sort_values(ascending=False)
        keep = list(strength.index[:max_cols])
        corr = corr.loc[keep, keep]

    n = corr.shape[0]
    size = max(4.5, min(9.0, 0.55 * n + 2.2))
    fig, ax = plt.subplots(figsize=(size, size * 0.86))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    # 대각선(항상 1)과 상삼각(하삼각과 중복)은 정보가 없으므로 잘라내고
    # 실제로 읽을 값이 있는 하삼각만 남긴다.
    full = np.array(corr.values, dtype=float)
    values = full[1:, :-1]
    row_labels = [_shorten(c, 14) for c in corr.columns[1:]]
    col_labels = [_shorten(c, 14) for c in corr.columns[:-1]]
    n_rows, n_cols = values.shape

    upper = np.array([[j > i for j in range(n_cols)] for i in range(n_rows)])
    cmap = DIV_CMAP.copy()
    cmap.set_bad("white")
    im = ax.imshow(np.ma.masked_where(upper, values), cmap=cmap, norm=norm)

    ax.set_xticks(range(n_cols), col_labels, rotation=45, ha="right")
    ax.set_yticks(range(n_rows), row_labels)
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="minor", color="white", linewidth=1.6)

    if n <= 12:  # 셀이 충분히 클 때만 값을 직접 표기
        for i in range(n_rows):
            for j in range(i + 1):
                value = values[i, j]
                if pd.isna(value):
                    continue
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if abs(value) > 0.55 else INK_SECONDARY,
                )

    cbar = fig.colorbar(im, ax=ax, shrink=0.72)
    cbar.set_label("")
    cbar.ax.set_title("상관계수", fontsize=8.5, color=INK_SECONDARY, pad=8)
    cbar.outline.set_edgecolor(AXIS_COLOR)
    ax.set_title("수치형 변수 상관계수 매트릭스")
    return Chart(
        key="correlation",
        title="수치형 변수 상관계수 매트릭스",
        path=_save(fig, out_dir, "correlation"),
        caption="파랑은 음(-)의 상관, 빨강은 양(+)의 상관을 나타내며 회색에 가까울수록 관계가 약합니다.",
    )


def plot_target_distribution(
    df: pd.DataFrame, target: str, task_type: str, out_dir: Path
) -> Optional[Chart]:
    """목표 변수의 분포(회귀=히스토그램, 분류=클래스별 막대)."""
    if target not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(CHART_FIGSIZE[0], 3.6))

    if task_type == "regression":
        values = pd.to_numeric(df[target], errors="coerce").dropna()
        if values.empty:
            plt.close(fig)
            return None
        ax.hist(values, bins=min(40, max(10, int(np.sqrt(len(values))))),
                color=PALETTE[0], edgecolor="white", linewidth=0.6)
        ax.set_xlabel(target)
        ax.set_ylabel("빈도")
        caption = "목표 변수의 분포입니다. 심한 편포가 보이면 변환 후 재학습을 검토하세요."
        _style_value_axis(ax, axis="y")
    else:
        counts = df[target].astype(str).value_counts().head(20)
        labels = [_shorten(i, 18) for i in counts.index]
        bars = ax.bar(labels, counts.values, color=PALETTE[0], width=0.62)
        total = counts.sum()
        for bar, value in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.008,
                f"{value:,}\n({value / total:.0%})",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=INK_SECONDARY,
            )
        ax.set_ylim(0, counts.max() * 1.22)
        ax.set_ylabel("건수")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ratio = counts.max() / max(counts.min(), 1)
        caption = (
            f"클래스 최대/최소 빈도 비율은 {ratio:.1f}배입니다. "
            "비율이 크면 정확도보다 F1 지표를 우선 확인하세요."
        )
        _style_value_axis(ax, axis="y")

    ax.set_title(f"목표 변수 '{target}' 분포")
    return Chart(
        key="target",
        title=f"목표 변수 '{target}' 분포",
        path=_save(fig, out_dir, "target_distribution"),
        caption=caption,
    )


def plot_boxplots(
    df: pd.DataFrame, profile: DataProfile, out_dir: Path, max_cols: int = 8
) -> Optional[Chart]:
    """수치형 변수 박스플롯(이상치 확인용, 표준화 후 동일 축에 배치)."""
    names = profile.numeric_columns[:max_cols]
    if len(names) < 2:
        return None

    data, labels = [], []
    for name in names:
        values = pd.to_numeric(df[name], errors="coerce").dropna()
        if values.empty or values.std(ddof=0) == 0:
            continue
        data.append(((values - values.mean()) / values.std(ddof=0)).to_numpy())
        labels.append(_shorten(name, 14))

    if len(data) < 2:
        return None

    fig, ax = plt.subplots(figsize=(CHART_FIGSIZE[0], 3.8))
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.55,
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": INK_MUTED,
                    "markeredgecolor": "none", "alpha": 0.5},
        medianprops={"color": "white", "linewidth": 1.6},
        whiskerprops={"color": AXIS_COLOR},
        capprops={"color": AXIS_COLOR},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE[0])
        patch.set_edgecolor("white")
        patch.set_linewidth(1.2)

    ax.set_title("수치형 변수 분포 비교 (표준화 기준)")
    ax.set_ylabel("표준화 값 (z-score)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _style_value_axis(ax, axis="y")
    return Chart(
        key="boxplot",
        title="수치형 변수 분포 비교",
        path=_save(fig, out_dir, "boxplot"),
        caption="모든 변수를 표준화하여 같은 축에서 비교했습니다. 상자 밖 점은 IQR 기준 이상치입니다.",
    )


# ---------------------------------------------------------------------------
# 2. 모델링 결과 차트
# ---------------------------------------------------------------------------
def plot_model_comparison(
    leaderboard: pd.DataFrame, metric: str, out_dir: Path
) -> Optional[Chart]:
    """후보 모델 성능 비교 막대(최적 모델만 강조색)."""
    if leaderboard.empty or metric not in leaderboard.columns:
        return None

    df = leaderboard.sort_values(metric, ascending=True)
    names = [_shorten(n, 18) for n in df["모델"]]
    values = df[metric].astype(float).to_numpy()

    height = max(2.8, 0.42 * len(df) + 1.2)
    fig, ax = plt.subplots(figsize=(CHART_FIGSIZE[0], height))
    colors = [INK_MUTED] * len(df)
    colors[-1] = PALETTE[0]  # 최고 성능 모델 강조
    bars = ax.barh(names, values, color=colors, height=0.6)

    span = max(abs(values.max()), 1e-9)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + span * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=8.5,
            color=INK_SECONDARY,
        )
    ax.set_xlim(min(0, values.min() * 1.1), span * 1.2)
    ax.set_xlabel(f"{metric} (교차검증 평균)")
    ax.set_title(f"후보 모델 성능 비교 — {metric} 기준")
    _pad_bar_axis(ax, len(df))
    _style_value_axis(ax, axis="x")
    return Chart(
        key="model_comparison",
        title="후보 모델 성능 비교",
        path=_save(fig, out_dir, "model_comparison"),
        caption="파란 막대가 최종 선택된 모델입니다. 값은 학습 데이터 교차검증 평균입니다.",
    )


def plot_feature_importance(
    importance: pd.DataFrame, out_dir: Path, top_n: int = 15
) -> Optional[Chart]:
    """변수 중요도 상위 N개 가로 막대."""
    if importance is None or importance.empty:
        return None

    # 기여도가 0인 변수는 막대가 보이지 않아 차트를 어지럽히므로 제외한다.
    positive = importance[importance["중요도"].astype(float) > 0]
    if positive.empty:
        return None
    n_zero = len(importance) - len(positive)

    df = positive.head(top_n).iloc[::-1]
    names = [_shorten(n, 24) for n in df["변수"]]
    values = df["중요도"].astype(float).to_numpy()
    if values.max() <= 0:
        return None

    height = max(2.8, 0.36 * len(df) + 1.2)
    fig, ax = plt.subplots(figsize=(CHART_FIGSIZE[0], height))
    bars = ax.barh(names, values, color=PALETTE[0], height=0.62)
    pct = df["중요도(%)"].astype(float).to_numpy()
    for bar, value, p in zip(bars, values, pct):
        ax.text(
            bar.get_width() + values.max() * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{p:.1f}%",
            va="center",
            fontsize=8.5,
            color=INK_SECONDARY,
        )
    ax.set_xlim(0, values.max() * 1.2)
    ax.set_xlabel("중요도 (성능 감소량)")
    ax.set_title(f"변수 중요도 상위 {len(df)}개")
    _pad_bar_axis(ax, len(df))
    _style_value_axis(ax, axis="x")

    caption = "해당 변수를 무작위로 섞었을 때 성능이 떨어지는 정도(순열 중요도)이며, 값이 클수록 결과에 큰 영향을 줍니다."
    if n_zero:
        caption += f" 기여도가 0으로 측정된 변수 {n_zero}개는 차트에서 제외했습니다."
    return Chart(
        key="importance",
        title="변수 중요도",
        path=_save(fig, out_dir, "feature_importance"),
        caption=caption,
    )


def plot_confusion_matrix(
    matrix: np.ndarray, labels: Sequence[str], out_dir: Path
) -> Optional[Chart]:
    """혼동 행렬(연속형 단일 색상 램프)."""
    if matrix is None or matrix.size == 0:
        return None

    n = matrix.shape[0]
    display = [_shorten(str(l), 12) for l in labels][:n]
    size = max(4.2, min(8.0, 0.7 * n + 3.0))
    fig, ax = plt.subplots(figsize=(size, size * 0.82))
    im = ax.imshow(matrix, cmap=SEQ_CMAP)

    ax.set_xticks(range(n), display, rotation=30, ha="right")
    ax.set_yticks(range(n), display)
    ax.set_xlabel("예측 클래스")
    ax.set_ylabel("실제 클래스")
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="minor", color="white", linewidth=1.6)

    threshold = matrix.max() * 0.55 if matrix.max() else 0
    for i in range(n):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:,}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if matrix[i, j] > threshold else INK_SECONDARY,
            )

    ax.set_title("혼동 행렬 (검증 데이터)")
    return Chart(
        key="confusion",
        title="혼동 행렬",
        path=_save(fig, out_dir, "confusion_matrix"),
        caption="대각선이 정확히 맞춘 건수입니다. 대각선 밖의 큰 값은 자주 혼동되는 클래스 쌍을 뜻합니다.",
    )


def plot_roc_curve(fpr, tpr, auc: float, out_dir: Path) -> Optional[Chart]:
    """이진 분류 ROC 곡선."""
    if fpr is None or tpr is None:
        return None

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.plot(fpr, tpr, color=PALETTE[0], linewidth=2.0, label=f"모델 (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1.2, linestyle="--", label="무작위 기준선")
    ax.set_xlabel("거짓 양성률 (FPR)")
    ax.set_ylabel("참 양성률 (TPR)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("ROC 곡선")
    ax.legend(loc="lower right")
    ax.grid(True)
    return Chart(
        key="roc",
        title="ROC 곡선",
        path=_save(fig, out_dir, "roc_curve"),
        caption="AUC가 1에 가까울수록 두 클래스를 잘 구분합니다. 0.5는 무작위 추측과 동일한 수준입니다.",
    )


def plot_regression_diagnostics(
    y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path
) -> List[Chart]:
    """실제-예측 산점도와 잔차 분포."""
    charts: List[Chart] = []
    if y_true is None or y_pred is None or len(y_true) == 0:
        return charts

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # (1) 실제 vs 예측
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.scatter(y_true, y_pred, s=26, color=PALETTE[0], alpha=0.6,
               edgecolors="white", linewidths=0.6)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color=INK_MUTED, linestyle="--", linewidth=1.2)
    ax.text(0.03, 0.95, "점선 = 완전 일치선", transform=ax.transAxes,
            fontsize=8.5, color=INK_SECONDARY, va="top")
    ax.set_xlabel("실제값")
    ax.set_ylabel("예측값")
    ax.set_title("실제값 대비 예측값")
    charts.append(
        Chart(
            key="pred_actual",
            title="실제값 대비 예측값",
            path=_save(fig, out_dir, "pred_vs_actual"),
            caption="점이 점선에 가까울수록 예측이 정확합니다. 특정 구간에서 체계적으로 벗어나면 해당 구간의 설명 변수를 보완해야 합니다.",
        )
    )

    # (2) 잔차 분포
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.hist(residuals, bins=min(40, max(10, int(np.sqrt(len(residuals))))),
            color=PALETTE[0], edgecolor="white", linewidth=0.6)
    ax.axvline(0, color=INK_SECONDARY, linestyle="--", linewidth=1.2)
    ax.set_xlabel("잔차 (실제값 − 예측값)")
    ax.set_ylabel("빈도")
    ax.set_title("잔차 분포")
    _style_value_axis(ax, axis="y")
    charts.append(
        Chart(
            key="residual",
            title="잔차 분포",
            path=_save(fig, out_dir, "residuals"),
            caption="잔차가 0을 중심으로 좌우 대칭이면 모델이 편향 없이 학습된 것으로 볼 수 있습니다.",
        )
    )
    return charts


def plot_target_by_category(
    df: pd.DataFrame,
    target: str,
    profile: DataProfile,
    out_dir: Path,
    max_charts: int = 2,
) -> List[Chart]:
    """범주형 변수별 목표 변수 평균(회귀 문제에서 실무 해석에 유용)."""
    charts: List[Chart] = []
    if target not in df.columns:
        return charts

    values = pd.to_numeric(df[target], errors="coerce")
    if values.notna().mean() < 0.8:
        return charts

    for name in profile.categorical_columns[:max_charts]:
        grouped = (
            df.assign(_t=values)
            .groupby(name, dropna=True)["_t"]
            .mean()
            .sort_values(ascending=False)
            .head(12)
        )
        if len(grouped) < 2:
            continue

        fig, ax = plt.subplots(figsize=(CHART_FIGSIZE[0], 3.6))
        labels = [_shorten(i, 16) for i in grouped.index]
        ax.bar(labels, grouped.values, color=PALETTE[0], width=0.62)
        overall = float(values.mean())
        ax.axhline(overall, color=INK_SECONDARY, linestyle="--", linewidth=1.2)
        ax.text(0.99, 0.95, f"전체 평균 {overall:,.4g}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5, color=INK_SECONDARY)
        ax.set_ylabel(f"{_shorten(target, 18)} 평균")
        ax.set_title(f"'{_shorten(name, 20)}' 구분별 {_shorten(target, 18)} 평균")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        _style_value_axis(ax, axis="y")
        charts.append(
            Chart(
                key="group_mean",
                title=f"{name} 구분별 평균",
                path=_save(fig, out_dir, f"group_mean_{len(charts)}"),
                caption="점선은 전체 평균입니다. 평균 차이가 큰 구분은 우선 관리 대상으로 볼 수 있습니다.",
            )
        )
    return charts


# ---------------------------------------------------------------------------
# 통합 진입점
# ---------------------------------------------------------------------------
def build_charts(
    df: pd.DataFrame,
    profile: DataProfile,
    out_dir: Path,
    *,
    target: Optional[str] = None,
    task_type: str = "eda",
    modeling=None,
    top_features: int = 15,
    primary_metric: str = "R2",
) -> ChartSet:
    """분석 결과 전체에 대한 차트를 한 번에 생성한다."""
    font, warning = init_style()
    result = ChartSet(font_name=font, font_warning=warning)

    # 탐색적 분석
    result.add(plot_missing_values(profile, out_dir))
    for chart in plot_numeric_distributions(df, profile, out_dir):
        result.add(chart)
    result.add(plot_boxplots(df, profile, out_dir))
    result.add(plot_correlation_heatmap(profile.correlation, out_dir))

    if target:
        result.add(plot_target_distribution(df, target, task_type, out_dir))
        if task_type == "regression":
            for chart in plot_target_by_category(df, target, profile, out_dir):
                result.add(chart)

    # 모델링
    if modeling is not None:
        result.add(plot_model_comparison(modeling.leaderboard, primary_metric, out_dir))
        result.add(plot_feature_importance(modeling.feature_importance, out_dir, top_features))

        if modeling.task_type == "classification":
            labels = modeling.class_labels or [
                str(i) for i in range(modeling.confusion.shape[0] if modeling.confusion is not None else 0)
            ]
            result.add(plot_confusion_matrix(modeling.confusion, labels, out_dir))
            if modeling.roc:
                fpr, tpr, auc = modeling.roc
                result.add(plot_roc_curve(fpr, tpr, auc, out_dir))
        else:
            for chart in plot_regression_diagnostics(modeling.y_test, modeling.y_pred, out_dir):
                result.add(chart)

    return result


def chart_index(chart_set: ChartSet) -> Dict[str, Chart]:
    """key → 첫 번째 차트 매핑(보고서 배치 시 사용)."""
    index: Dict[str, Chart] = {}
    for chart in chart_set.charts:
        index.setdefault(chart.key, chart)
    return index
