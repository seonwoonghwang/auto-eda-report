"""전역 설정 및 상수 정의 모듈.

애플리케이션 전반에서 사용하는 경로, 기본 파라미터, 한글 폰트 설정을 담당한다.
외부 LLM API에 전혀 의존하지 않는 순수 로컬 환경을 전제로 한다.
"""

from __future__ import annotations

import os
import platform
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# --------------------------------------------------------------------------
# 경로
# --------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def ensure_dir(path: Path) -> Path:
    """디렉터리가 없으면 생성하고 경로를 반환한다."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_workspace(prefix: str = "autoeda_") -> Path:
    """차트 이미지 등 임시 산출물을 담을 작업 디렉터리를 생성한다."""
    return Path(tempfile.mkdtemp(prefix=prefix))


def is_writable(path: Path) -> bool:
    """해당 경로에 실제로 파일을 쓸 수 있는지 확인한다."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def writable_output_dir() -> Path:
    """보고서를 저장할 수 있는 폴더를 반환한다.

    클라우드 배포처럼 프로젝트 폴더가 읽기 전용인 환경에서는 임시 폴더로 대체하여,
    보고서 생성이 실패하지 않도록 한다(다운로드 버튼으로는 동일하게 받을 수 있다).
    """
    if is_writable(DEFAULT_OUTPUT_DIR):
        return DEFAULT_OUTPUT_DIR
    return ensure_dir(Path(tempfile.gettempdir()) / "autoeda_outputs")


# --------------------------------------------------------------------------
# 데이터 처리 기본값
# --------------------------------------------------------------------------
#: CSV 읽기 시 순서대로 시도할 인코딩 후보
ENCODING_CANDIDATES: List[str] = [
    "utf-8-sig",
    "utf-8",
    "cp949",
    "euc-kr",
    "latin1",
]

#: CSV 구분자 후보 (None 이면 pandas 자동 추론 시도)
SEPARATOR_CANDIDATES: List[Optional[str]] = [None, ",", ";", "\t", "|"]

#: 고유값 개수가 이 값 이하인 정수/문자 컬럼은 범주형으로 추정한다.
CATEGORICAL_UNIQUE_THRESHOLD = 20

#: 고유값 비율이 이 값 이상인 문자열 컬럼은 식별자(ID)/자유 텍스트로 간주하고 제외 후보로 둔다.
HIGH_CARDINALITY_RATIO = 0.5

#: 결측 비율이 이 값을 초과하는 컬럼은 자동 제외 후보로 표시한다.
MISSING_DROP_RATIO = 0.6

#: 분류 문제로 판정하기 위한 목표 변수의 최대 고유값 수(수치형 목표 변수 기준)
CLASSIFICATION_MAX_CLASSES = 20


# --------------------------------------------------------------------------
# 시각화 설정
# --------------------------------------------------------------------------
CHART_DPI = 150
CHART_FIGSIZE = (8.0, 4.8)

#: 범주형(계열 구분) 팔레트. 색각 이상 대비를 검증한 고정 순서이며 순환 사용하지 않는다.
PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

#: 연속형(크기) 인코딩용 단일 색상 램프 — 밝음 → 어두움
SEQUENTIAL_RAMP = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#184f95",
    "#0d366b",
]

#: 발산형(양/음 극성) 인코딩용 — 파랑 ↔ 회색 ↔ 빨강
DIVERGING_LOW = "#184f95"
DIVERGING_MID = "#f0efec"
DIVERGING_HIGH = "#c0392f"

#: 차트 크롬(잉크) 색상
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
SURFACE = "#ffffff"

#: 운영체제별 한글 폰트 후보. 앞에서부터 설치 여부를 확인해 사용한다.
KOREAN_FONT_CANDIDATES = [
    "Malgun Gothic",      # Windows 기본
    "NanumGothic",        # Linux/Windows 나눔고딕
    "NanumBarunGothic",
    "AppleGothic",        # macOS
    "Noto Sans CJK KR",
    "Noto Sans KR",
    "DejaVu Sans",        # 최종 폴백(한글 미지원, 경고 표시)
]

_FONT_CACHE: dict = {}


def resolve_korean_font() -> str:
    """설치된 한글 폰트 이름을 찾아 반환한다. 없으면 폴백 폰트명을 반환."""
    if "name" in _FONT_CACHE:
        return _FONT_CACHE["name"]

    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next(
        (name for name in KOREAN_FONT_CANDIDATES if name in installed),
        "DejaVu Sans",
    )
    _FONT_CACHE["name"] = chosen
    return chosen


def apply_matplotlib_style() -> str:
    """matplotlib 전역 스타일과 한글 폰트를 적용하고 사용된 폰트명을 반환한다."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    font_name = resolve_korean_font()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "axes.unicode_minus": False,
            "figure.dpi": CHART_DPI,
            "savefig.dpi": CHART_DPI,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": AXIS_COLOR,
            "axes.linewidth": 1.0,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "axes.labelcolor": INK_SECONDARY,
            "text.color": INK_PRIMARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.bottom": True,
            "ytick.left": False,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )
    return font_name


# --------------------------------------------------------------------------
# 분석 설정 객체
# --------------------------------------------------------------------------
@dataclass
class AnalysisConfig:
    """사용자가 GUI에서 선택한 분석 조건을 담는 객체."""

    #: "eda" | "regression" | "classification" | "auto"
    analysis_type: str = "auto"
    #: 목표 변수(타깃) 컬럼명. EDA 전용 분석이면 None.
    target: Optional[str] = None
    #: 분석에서 제외할 컬럼 목록
    exclude_columns: List[str] = field(default_factory=list)
    #: 사용자가 수동 지정한 컬럼 타입 {컬럼명: "numeric"|"categorical"|"datetime"|"exclude"}
    column_types: dict = field(default_factory=dict)

    # 전처리 옵션
    numeric_impute: str = "median"        # mean | median | constant
    categorical_impute: str = "most_frequent"  # most_frequent | constant
    handle_outliers: bool = True
    outlier_method: str = "iqr"           # iqr | none
    scaling: str = "standard"             # standard | minmax | none

    # 모델링 옵션
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    max_models: int = 8
    compute_importance: bool = True

    # 보고서 옵션
    report_title: str = "데이터 분석 자동화 보고서"
    report_subtitle: str = ""
    author: str = ""
    organization: str = ""
    include_raw_preview: bool = True
    top_features: int = 15

    def resolved_output_dir(self, base: Optional[os.PathLike] = None) -> Path:
        if base:
            candidate = Path(base)
            if is_writable(candidate):
                return candidate
        return writable_output_dir()


def system_summary() -> dict:
    """실행 환경 요약 (보고서 부록에 기재)."""
    return {
        "운영체제": f"{platform.system()} {platform.release()}",
        "파이썬": platform.python_version(),
    }
