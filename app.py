"""Standalone Auto-EDA & Report Builder — Streamlit GUI.

실행:

    streamlit run app.py

코딩 없이 마우스 조작만으로 데이터 업로드 → 분석 설정 → 자동 분석 →
Word 보고서 다운로드까지 수행한다.
"""

from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from autoeda import __version__  # noqa: E402
from autoeda.config import AnalysisConfig, writable_output_dir  # noqa: E402
from autoeda.data_loader import DataLoadError, list_excel_sheets, load_dataframe  # noqa: E402
from autoeda.pipeline import run_analysis  # noqa: E402
from autoeda.profiling import (  # noqa: E402
    CATEGORICAL,
    DATETIME,
    NUMERIC,
    TEXT,
    profile_dataframe,
    suggest_task_type,
)

TYPE_LABELS = {
    NUMERIC: "수치형",
    CATEGORICAL: "범주형",
    DATETIME: "날짜형",
    TEXT: "텍스트/식별자(제외)",
}
LABEL_TO_TYPE = {v: k for k, v in TYPE_LABELS.items()}

st.set_page_config(
    page_title="데이터 분석 자동화 도구",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
  .block-container { padding-top: 2.2rem; max-width: 1180px; }
  h1, h2, h3 { letter-spacing: -0.01em; }
  div[data-testid="stMetricValue"] { font-size: 1.35rem; }
  .step-badge {
      display:inline-block; background:#1C5CAB; color:#fff; border-radius:999px;
      padding:2px 11px; font-size:0.78rem; font-weight:600; margin-right:8px;
  }
  .hint { color:#52514E; font-size:0.86rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 세션 상태
# ---------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "df": None,
        "load_result": None,
        "profile": None,
        "result": None,
        "type_overrides": {},
        "error": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()


def step_header(number: int, title: str, hint: str = "") -> None:
    st.markdown(
        f'<h3><span class="step-badge">STEP {number}</span>{title}</h3>'
        + (f'<p class="hint">{hint}</p>' if hint else ""),
        unsafe_allow_html=True,
    )


def reset_analysis() -> None:
    st.session_state.result = None


# ---------------------------------------------------------------------------
# 사이드바 — 고급 설정
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ 분석 옵션")
    st.caption("기본값 그대로 두어도 정상 동작합니다.")

    with st.expander("전처리", expanded=False):
        numeric_impute = st.selectbox(
            "수치형 결측 대체", ["median", "mean", "constant"],
            format_func=lambda x: {"median": "중앙값", "mean": "평균", "constant": "0으로 채움"}[x],
        )
        categorical_impute = st.selectbox(
            "범주형 결측 대체", ["most_frequent", "constant"],
            format_func=lambda x: {"most_frequent": "최빈값", "constant": "'미상'으로 채움"}[x],
        )
        handle_outliers = st.checkbox("이상치 자동 보정 (IQR 윈저라이징)", value=True)
        scaling = st.selectbox(
            "스케일링", ["standard", "minmax", "none"],
            format_func=lambda x: {"standard": "표준화", "minmax": "정규화(0~1)", "none": "사용 안 함"}[x],
        )

    with st.expander("모델링", expanded=False):
        test_size = st.slider("검증 데이터 비율", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.slider("교차검증 폴드 수", 2, 10, 5)
        max_models = st.slider("비교할 알고리즘 수", 2, 11, 8)
        compute_importance = st.checkbox("변수 중요도 계산", value=True)
        random_state = st.number_input("난수 시드", value=42, step=1)

    with st.expander("보고서", expanded=False):
        report_title = st.text_input("보고서 제목", value="데이터 분석 자동화 보고서")
        report_subtitle = st.text_input("부제 (선택)", value="")
        organization = st.text_input("소속 (선택)", value="")
        author = st.text_input("작성자 (선택)", value="")
        top_features = st.slider("보고서에 실을 변수 중요도 개수", 5, 30, 15)
        include_raw_preview = st.checkbox("원본 데이터 미리보기 포함", value=True)
        output_dir = st.text_input(
            "저장 폴더",
            value=str(writable_output_dir()),
            help="쓰기 권한이 없는 경로를 입력하면 임시 폴더로 자동 대체됩니다.",
        )

    st.divider()
    st.caption(f"Auto-EDA & Report Builder v{__version__}\n\n외부 서버 전송 없이 로컬에서만 동작합니다.")


# ---------------------------------------------------------------------------
# 헤더
# ---------------------------------------------------------------------------
st.title("📊 데이터 분석 자동화 도구")
st.markdown(
    '<p class="hint">CSV·Excel 파일을 올리고 목표 변수만 고르면 전처리·통계 분석·예측 모델링·'
    'Word 보고서까지 자동으로 만들어 드립니다. 코딩은 필요하지 않습니다.</p>',
    unsafe_allow_html=True,
)
st.divider()


# ---------------------------------------------------------------------------
# STEP 1 — 데이터 업로드
# ---------------------------------------------------------------------------
step_header(1, "데이터 업로드", "CSV, XLSX, XLS 파일을 지원합니다. 한글 인코딩(cp949/euc-kr)은 자동 인식합니다.")

col_upload, col_sample = st.columns([3, 1])

with col_upload:
    uploaded = st.file_uploader(
        "파일을 끌어다 놓거나 클릭해서 선택하세요",
        type=["csv", "txt", "tsv", "xlsx", "xls", "xlsm"],
        label_visibility="collapsed",
    )

with col_sample:
    sample_dir = Path(__file__).resolve().parent / "samples"
    sample_meta = [
        ("sample_quality_classification.csv", "🏭 제조 품질 판정 (분류)"),
        ("sample_sales_regression.csv", "🏬 지점 매출 예측 (회귀)"),
        ("sample_card_customer_churn.csv", "💳 카드사 고객 이탈 (분류)"),
    ]
    available_samples = [
        (sample_dir / name, label) for name, label in sample_meta if (sample_dir / name).exists()
    ]
    if available_samples:
        chosen_label = st.selectbox(
            "샘플 데이터 선택",
            [label for _, label in available_samples],
            label_visibility="collapsed",
        )
        chosen_path = next(path for path, label in available_samples if label == chosen_label)
        if st.button("샘플 데이터로 체험", use_container_width=True):
            loaded = load_dataframe(chosen_path)
            st.session_state.df = loaded.df
            st.session_state.load_result = loaded
            st.session_state.profile = profile_dataframe(loaded.df)
            st.session_state.type_overrides = {}
            reset_analysis()
            st.rerun()

if uploaded is not None:
    try:
        raw = uploaded.getvalue()
        sheet: Optional[str] = None
        if Path(uploaded.name).suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
            sheets = list_excel_sheets(raw)
            if len(sheets) > 1:
                sheet = st.selectbox("시트 선택", sheets)

        signature = (uploaded.name, len(raw), sheet)
        if st.session_state.get("_signature") != signature:
            loaded = load_dataframe(raw, filename=uploaded.name, sheet_name=sheet)
            st.session_state.df = loaded.df
            st.session_state.load_result = loaded
            st.session_state.profile = profile_dataframe(loaded.df)
            st.session_state.type_overrides = {}
            st.session_state["_signature"] = signature
            st.session_state.error = ""
            reset_analysis()
    except DataLoadError as exc:
        st.session_state.error = str(exc)
    except Exception as exc:  # noqa: BLE001
        st.session_state.error = f"파일을 읽는 중 오류가 발생했습니다: {exc}"

if st.session_state.error:
    st.error(st.session_state.error)

df: Optional[pd.DataFrame] = st.session_state.df

if df is None:
    st.info("분석할 파일을 업로드하면 다음 단계가 나타납니다.")
    st.stop()

loaded = st.session_state.load_result
profile = st.session_state.profile

m1, m2, m3, m4 = st.columns(4)
m1.metric("행 수", f"{profile.n_rows:,}")
m2.metric("열 수", f"{profile.n_cols:,}")
m3.metric("결측 컬럼", f"{sum(1 for c in profile.columns.values() if c.n_missing):,}")
m4.metric("중복 행", f"{profile.n_duplicated_rows:,}")

for note in (loaded.notes if loaded else []):
    st.caption(f"ℹ️ {note}")

with st.expander("데이터 미리보기", expanded=True):
    st.dataframe(df.head(50), use_container_width=True, height=280)

with st.expander("컬럼 요약 보기"):
    st.dataframe(profile.overview_table(), use_container_width=True, hide_index=True)

st.divider()


# ---------------------------------------------------------------------------
# STEP 2 — 컬럼 타입 확인
# ---------------------------------------------------------------------------
step_header(2, "컬럼 타입 확인", "자동 인식 결과가 맞지 않으면 여기서 바꿀 수 있습니다.")

type_editor = pd.DataFrame(
    [
        {
            "컬럼명": name,
            "타입": TYPE_LABELS[cp.inferred_type],
            "결측률(%)": round(cp.missing_ratio * 100, 1),
            "고유값 수": cp.n_unique,
            "예시": ", ".join(cp.sample_values[:3]),
        }
        for name, cp in profile.columns.items()
    ]
)

edited = st.data_editor(
    type_editor,
    use_container_width=True,
    hide_index=True,
    height=min(420, 45 + 35 * len(type_editor)),
    disabled=["컬럼명", "결측률(%)", "고유값 수", "예시"],
    column_config={
        "타입": st.column_config.SelectboxColumn(
            "타입", options=list(TYPE_LABELS.values()), required=True
        )
    },
    key="type_editor",
)

overrides = {
    row["컬럼명"]: LABEL_TO_TYPE[row["타입"]]
    for _, row in edited.iterrows()
    if LABEL_TO_TYPE[row["타입"]] != profile.columns[row["컬럼명"]].inferred_type
}
st.session_state.type_overrides = overrides
if overrides:
    st.caption(f"✏️ {len(overrides)}개 컬럼의 타입을 수동 지정했습니다.")

st.divider()


# ---------------------------------------------------------------------------
# STEP 3 — 분석 설정
# ---------------------------------------------------------------------------
step_header(3, "분석 설정", "무엇을 예측할지 고르세요. 목표 변수를 비워 두면 탐색적 분석만 수행합니다.")

col_target, col_type = st.columns([2, 2])

with col_target:
    target_options = ["(선택 안 함 — 탐색적 분석만)"] + list(df.columns)
    target_choice = st.selectbox("🎯 목표 변수 (예측하려는 항목)", target_options)
    target = None if target_choice == target_options[0] else target_choice

with col_type:
    suggested = suggest_task_type(df, target) if target else "eda"
    suggestion_label = {
        "regression": "회귀 예측 (수치 예측)",
        "classification": "분류 예측 (범주 판별)",
        "eda": "탐색적 분석",
    }[suggested]
    analysis_choice = st.selectbox(
        "📈 분석 유형",
        ["auto", "regression", "classification", "eda"],
        format_func=lambda x: {
            "auto": f"자동 판정 (추천: {suggestion_label})",
            "regression": "회귀 예측 (수치 예측)",
            "classification": "분류 예측 (범주 판별)",
            "eda": "탐색적 분석만",
        }[x],
    )

auto_exclude = [c for c in profile.suggested_exclusions if c != target]
exclude_columns = st.multiselect(
    "🚫 분석에서 제외할 컬럼",
    [c for c in df.columns if c != target],
    default=auto_exclude,
    help="식별번호, 이름처럼 예측에 도움이 되지 않는 컬럼을 제외하세요. 자동 추천 항목이 미리 선택되어 있습니다.",
)

if target:
    used = len(df.columns) - len(exclude_columns) - 1
    st.caption(f"📌 목표 변수 **{target}** / 설명 변수 **{used}개** 로 분석합니다.")

st.divider()


# ---------------------------------------------------------------------------
# STEP 4 — 분석 실행
# ---------------------------------------------------------------------------
step_header(4, "분석 실행", "전처리 → 모델 비교 → 차트 생성 → Word 보고서 작성이 한 번에 진행됩니다.")

run_clicked = st.button("🚀 분석 시작", type="primary", use_container_width=True)

if run_clicked:
    config = AnalysisConfig(
        analysis_type=analysis_choice,
        target=target,
        exclude_columns=exclude_columns,
        column_types=st.session_state.type_overrides,
        numeric_impute=numeric_impute,
        categorical_impute=categorical_impute,
        handle_outliers=handle_outliers,
        scaling=scaling,
        test_size=test_size,
        cv_folds=cv_folds,
        max_models=max_models,
        compute_importance=compute_importance,
        random_state=int(random_state),
        report_title=report_title,
        report_subtitle=report_subtitle,
        organization=organization,
        author=author,
        top_features=top_features,
        include_raw_preview=include_raw_preview,
    )

    bar = st.progress(0.0, text="분석을 시작합니다…")

    def on_progress(ratio: float, message: str) -> None:
        bar.progress(ratio, text=message)

    try:
        with st.spinner("분석 중입니다. 데이터 크기에 따라 수십 초가 걸릴 수 있습니다…"):
            result = run_analysis(
                df,
                config,
                source_meta=loaded.meta_dict() if loaded else {},
                output_dir=Path(output_dir),
                progress=on_progress,
                load_notes=loaded.notes if loaded else None,
            )
        st.session_state.result = result
        bar.progress(1.0, text="완료")
    except Exception as exc:  # noqa: BLE001
        bar.empty()
        st.error(f"분석 중 오류가 발생했습니다: {exc}")
        with st.expander("자세한 오류 내용"):
            st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# STEP 5 — 결과 확인 및 다운로드
# ---------------------------------------------------------------------------
result = st.session_state.result
if result is None:
    st.stop()

st.divider()
step_header(5, "결과 확인 및 보고서 다운로드")

for message in result.messages:
    st.info(message)

if result.report_path and Path(result.report_path).exists():
    report_path = Path(result.report_path)
    st.success(f"보고서가 생성되었습니다 — {report_path}")
    with open(report_path, "rb") as handle:
        st.download_button(
            "📥 Word 보고서 내려받기 (.docx)",
            data=handle.read(),
            file_name=report_path.name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary",
            use_container_width=True,
        )

tabs = st.tabs(["📈 모델 성능", "🔑 주요 변수", "🖼 차트", "🧹 전처리 내역", "📋 기초 통계"])

with tabs[0]:
    if result.modeling is None:
        st.info("목표 변수를 지정하지 않아 모델링을 수행하지 않았습니다.")
    else:
        m = result.modeling
        st.markdown(f"#### 최종 선택 모델 — **{m.best_model_name}**")
        metrics = list(m.holdout_metrics.items())[:4]
        cols = st.columns(len(metrics))
        for col, (key, value) in zip(cols, metrics):
            col.metric(key, f"{value:,.4g}" if abs(value) >= 0.01 else f"{value:.4f}")
        st.markdown("##### 후보 모델 비교 (교차검증 평균)")
        st.dataframe(m.leaderboard_display(), use_container_width=True, hide_index=True)
        st.markdown("##### 검증 데이터 성능")
        st.dataframe(m.holdout_table(), use_container_width=True, hide_index=True)

with tabs[1]:
    if result.modeling is None or result.modeling.feature_importance.empty:
        st.info("변수 중요도 정보가 없습니다.")
    else:
        st.dataframe(
            result.modeling.feature_importance.head(top_features),
            use_container_width=True,
            hide_index=True,
        )

with tabs[2]:
    if not result.charts.charts:
        st.info("생성된 차트가 없습니다.")
    for chart in result.charts.charts:
        st.markdown(f"**{chart.title}**")
        st.image(str(chart.path), use_container_width=True)
        if chart.caption:
            st.caption(chart.caption)

with tabs[3]:
    st.dataframe(
        result.prepared.report.to_table(), use_container_width=True, hide_index=True
    )

with tabs[4]:
    describe = result.profile.numeric_describe_table()
    if not describe.empty:
        st.markdown("##### 수치형 기술통계")
        st.dataframe(describe, use_container_width=True, hide_index=True)
    categorical = result.profile.categorical_table()
    if not categorical.empty:
        st.markdown("##### 범주형 요약")
        st.dataframe(categorical, use_container_width=True, hide_index=True)
    if result.profile.correlation is not None:
        st.markdown("##### 상관계수 매트릭스")
        st.dataframe(
            result.profile.correlation.round(3), use_container_width=True
        )
