"""Standalone Auto-EDA & Report Builder.

외부 LLM API 없이 로컬에서 동작하는 No-Code 데이터 분석·보고서 자동화 엔진.

기본 사용법::

    from autoeda import AnalysisConfig, run_from_file

    config = AnalysisConfig(target="매출액", analysis_type="auto")
    result = run_from_file("sales.csv", config)
    print(result.report_path)
"""

from .automl import ModelingResult, run_automl
from .config import AnalysisConfig
from .data_loader import DataLoadError, LoadResult, load_dataframe
from .pipeline import AnalysisResult, run_analysis, run_from_file
from .preprocessing import PreparedData, prepare_data
from .profiling import DataProfile, profile_dataframe, suggest_task_type
from .report_builder import build_report, default_report_name
from .visualization import ChartSet, build_charts

__version__ = "1.0.0"
__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "ChartSet",
    "DataLoadError",
    "DataProfile",
    "LoadResult",
    "ModelingResult",
    "PreparedData",
    "build_charts",
    "build_report",
    "default_report_name",
    "load_dataframe",
    "prepare_data",
    "profile_dataframe",
    "run_analysis",
    "run_automl",
    "run_from_file",
    "suggest_task_type",
]
