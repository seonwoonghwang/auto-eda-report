"""분석 전체 흐름을 묶는 오케스트레이터.

GUI(Streamlit)와 CLI 모두 이 모듈의 `run_analysis` 하나만 호출하면 된다.

    적재 → 프로파일링 → 전처리 → AutoML → 차트 → Word 보고서
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from .automl import PRIMARY_METRIC, ModelingResult, run_automl
from .config import AnalysisConfig, new_workspace
from .data_loader import LoadResult, load_dataframe
from .preprocessing import PreparedData, prepare_data
from .profiling import DataProfile, profile_dataframe, suggest_task_type
from .report_builder import build_report, default_report_name
from .visualization import ChartSet, build_charts

ProgressFn = Optional[Callable[[float, str], None]]


@dataclass
class AnalysisResult:
    """분석 실행 결과 전체."""

    profile: DataProfile
    prepared: PreparedData
    charts: ChartSet
    task_type: str
    report_path: Optional[Path] = None
    modeling: Optional[ModelingResult] = None
    source_meta: Dict[str, str] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
    workspace: Optional[Path] = None

    @property
    def has_model(self) -> bool:
        return self.modeling is not None


def _notify(progress: ProgressFn, ratio: float, message: str) -> None:
    if progress:
        try:
            progress(min(max(ratio, 0.0), 1.0), message)
        except Exception:  # noqa: BLE001 - 진행률 콜백 오류는 분석을 막지 않는다
            pass


def run_analysis(
    df: pd.DataFrame,
    config: AnalysisConfig,
    *,
    source_meta: Optional[Dict[str, str]] = None,
    output_dir: Optional[Path] = None,
    build_docx: bool = True,
    progress: ProgressFn = None,
    load_notes: Optional[List[str]] = None,
) -> AnalysisResult:
    """DataFrame 하나로부터 분석 → 보고서까지 수행한다."""
    messages: List[str] = []
    workspace = new_workspace()
    # 지정된 폴더에 쓸 수 없으면(클라우드의 읽기 전용 경로 등) 임시 폴더로 자동 대체된다.
    out_dir = config.resolved_output_dir(output_dir)

    # 1) 프로파일링 --------------------------------------------------------
    _notify(progress, 0.05, "데이터 프로파일링 중…")
    profile = profile_dataframe(df, overrides=config.column_types or None)

    # 2) 문제 유형 결정 ----------------------------------------------------
    task_type = config.analysis_type
    if task_type in ("auto", ""):
        task_type = suggest_task_type(df, config.target)
        if config.target:
            messages.append(
                f"목표 변수 '{config.target}'의 특성을 근거로 "
                f"{'회귀' if task_type == 'regression' else '분류' if task_type == 'classification' else '탐색적'} "
                "분석으로 자동 판정했습니다."
            )
    if not config.target:
        task_type = "eda"

    # 3) 전처리 ------------------------------------------------------------
    _notify(progress, 0.15, "자동 전처리 수행 중…")
    prepared = prepare_data(df, profile, config, task_type)

    # 4) 모델링 ------------------------------------------------------------
    modeling: Optional[ModelingResult] = None
    if task_type in ("regression", "classification"):
        if prepared.X.empty or prepared.X.shape[1] == 0:
            messages.append("사용 가능한 설명 변수가 없어 모델링을 건너뛰고 EDA만 수행했습니다.")
            task_type = "eda"
        else:
            def model_progress(ratio: float, text: str) -> None:
                _notify(progress, 0.2 + ratio * 0.5, text)

            try:
                modeling = run_automl(prepared, config, task_type, progress=model_progress)
                messages.extend(modeling.warnings_)
            except Exception as exc:  # noqa: BLE001 - 모델링 실패 시 EDA 로 폴백
                messages.append(f"모델링을 완료하지 못해 탐색적 분석만 수행했습니다. (원인: {exc})")
                task_type = "eda"

    # 5) 차트 --------------------------------------------------------------
    _notify(progress, 0.75, "차트 생성 중…")
    charts = build_charts(
        df,
        profile,
        workspace,
        target=config.target,
        task_type=task_type,
        modeling=modeling,
        top_features=config.top_features,
        primary_metric=PRIMARY_METRIC.get(task_type, "R2"),
    )
    if charts.font_warning:
        messages.append(charts.font_warning)

    # 6) 보고서 ------------------------------------------------------------
    report_path: Optional[Path] = None
    if build_docx:
        _notify(progress, 0.9, "Word 보고서 작성 중…")
        report_path = build_report(
            output_path=out_dir / default_report_name(),
            config=config,
            profile=profile,
            raw_preview=df.head(10),
            source_meta=source_meta or {},
            preprocessing_table=prepared.report.to_table(),
            charts=charts,
            modeling=modeling,
            load_notes=load_notes,
        )

    _notify(progress, 1.0, "완료")
    return AnalysisResult(
        profile=profile,
        prepared=prepared,
        charts=charts,
        task_type=task_type,
        report_path=report_path,
        modeling=modeling,
        source_meta=source_meta or {},
        messages=messages,
        workspace=workspace,
    )


def run_from_file(
    path: Path,
    config: AnalysisConfig,
    *,
    sheet_name: Optional[str] = None,
    output_dir: Optional[Path] = None,
    progress: ProgressFn = None,
) -> AnalysisResult:
    """파일 경로에서 곧바로 전체 분석을 수행한다(CLI 진입점용)."""
    loaded: LoadResult = load_dataframe(path, sheet_name=sheet_name)
    return run_analysis(
        loaded.df,
        config,
        source_meta=loaded.meta_dict(),
        output_dir=output_dir,
        progress=progress,
        load_notes=loaded.notes,
    )
