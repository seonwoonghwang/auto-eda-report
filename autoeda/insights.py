"""분석 결과를 한국어 문장으로 요약하는 규칙 기반 해석 모듈.

외부 LLM을 사용하지 않고, 통계값의 임계치에 따른 사전 정의 문장 템플릿으로
'분석 요약'과 '시사점'을 생성한다.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .automl import ModelingResult
from .profiling import DataProfile, top_correlations


def data_summary_sentences(profile: DataProfile, source_name: str) -> List[str]:
    """데이터 개요 문단."""
    out = [
        f"분석 대상 데이터는 '{source_name}' 파일로, 총 {profile.n_rows:,}건의 관측치와 "
        f"{profile.n_cols:,}개의 컬럼으로 구성되어 있습니다."
    ]
    out.append(
        f"컬럼 구성은 수치형 {len(profile.numeric_columns)}개, 범주형 "
        f"{len(profile.categorical_columns)}개, 날짜형 {len(profile.datetime_columns)}개, "
        f"텍스트/식별자 {len(profile.text_columns)}개입니다."
    )

    missing_cols = [c for c in profile.columns.values() if c.n_missing > 0]
    if not missing_cols:
        out.append("결측값이 있는 컬럼은 없어 별도의 결측 대체 없이 분석이 가능했습니다.")
    else:
        worst = max(missing_cols, key=lambda c: c.missing_ratio)
        out.append(
            f"결측값이 존재하는 컬럼은 {len(missing_cols)}개이며, 결측률이 가장 높은 컬럼은 "
            f"'{worst.name}'({worst.missing_ratio:.1%})입니다."
        )

    if profile.n_duplicated_rows:
        out.append(
            f"완전히 동일한 중복 행이 {profile.n_duplicated_rows:,}건 발견되어 데이터 수집 과정의 "
            "중복 입력 여부를 확인할 필요가 있습니다."
        )

    heavy_outliers = sorted(
        (c for c in profile.columns.values() if c.outlier_ratio > 0.05),
        key=lambda c: c.outlier_ratio,
        reverse=True,
    )[:3]
    if heavy_outliers:
        names = ", ".join(f"'{c.name}'({c.outlier_ratio:.1%})" for c in heavy_outliers)
        out.append(f"IQR 기준 이상치 비율이 5%를 넘는 컬럼은 {names} 입니다.")

    return out


def correlation_sentences(profile: DataProfile) -> List[str]:
    """상관관계 해석 문단."""
    pairs = top_correlations(profile.correlation, top_n=3)
    if pairs.empty:
        return ["수치형 변수가 부족하여 상관관계 분석은 수행하지 않았습니다."]

    out: List[str] = []
    strongest = pairs.iloc[0]
    direction = "양(+)" if strongest["상관계수"] > 0 else "음(−)"
    out.append(
        f"가장 강한 상관을 보인 변수쌍은 '{strongest['변수 1']}'와 '{strongest['변수 2']}'로, "
        f"상관계수 {strongest['상관계수']:.3f}의 {direction}의 관계입니다."
    )

    high = pairs[pairs["상관계수"].abs() >= 0.85]
    if not high.empty:
        out.append(
            f"상관계수 절댓값이 0.85 이상인 변수쌍이 {len(high)}쌍 존재하여 다중공선성이 우려됩니다. "
            "선형 모델을 사용할 경우 중복 변수 제거를 검토하십시오."
        )
    elif pairs["상관계수"].abs().max() < 0.3:
        out.append(
            "전반적으로 변수 간 선형 상관이 약해, 단순 선형 관계보다 비선형 모델이 유리할 수 있습니다."
        )
    return out


def modeling_sentences(result: ModelingResult) -> List[str]:
    """모델링 결과 해석 문단."""
    task_kor = "회귀(수치 예측)" if result.task_type == "regression" else "분류(범주 예측)"
    out = [
        f"목표 변수 '{result.target}'에 대해 {task_kor} 문제로 판정하고, "
        f"총 {len(result.leaderboard)}개 알고리즘을 교차검증으로 비교했습니다."
    ]
    out.append(
        f"최종 선택된 모델은 **{result.best_model_name}**이며, "
        f"학습 {result.n_train:,}건 / 검증 {result.n_test:,}건으로 평가했습니다."
    )

    m = result.holdout_metrics
    if result.task_type == "regression":
        r2 = m.get("R2", 0.0)
        out.append(
            f"검증 데이터 기준 결정계수(R²)는 {r2:.3f}, RMSE는 {m.get('RMSE', 0):,.4g}, "
            f"MAE는 {m.get('MAE', 0):,.4g} 입니다."
        )
        if r2 >= 0.8:
            out.append("설명력이 높아 실무 예측에 활용 가능한 수준입니다.")
        elif r2 >= 0.5:
            out.append(
                "설명력은 중간 수준입니다. 추가 변수 확보나 파생 변수 설계로 개선 여지가 있습니다."
            )
        elif r2 >= 0.2:
            out.append(
                "설명력이 낮은 편으로, 현재 변수만으로는 목표 변수의 변동을 충분히 설명하기 어렵습니다."
            )
        else:
            out.append(
                "설명력이 매우 낮습니다. 목표 변수 정의나 데이터 수집 범위를 재검토할 필요가 있습니다."
            )
    else:
        acc = m.get("정확도", 0.0)
        f1 = m.get("F1(가중)", 0.0)
        out.append(f"검증 데이터 기준 정확도는 {acc:.1%}, 가중 F1 점수는 {f1:.3f} 입니다.")
        if "ROC-AUC" in m:
            out.append(f"이진 분류 ROC-AUC는 {m['ROC-AUC']:.3f} 입니다.")
        gap = m.get("기준모델 대비 정확도 개선율")
        if gap is not None and gap < 5:
            out.append(
                "최빈 클래스만 예측하는 기준 모델 대비 개선폭이 작아, 현재 변수의 설명력이 제한적입니다."
            )
        if f1 >= 0.85:
            out.append("클래스 전반에 걸쳐 안정적인 성능을 보입니다.")
        elif f1 < 0.6:
            out.append(
                "성능이 낮은 편이므로 클래스 불균형 보정이나 추가 변수 확보를 검토하십시오."
            )

    return out


def importance_sentences(result: ModelingResult, top_n: int = 5) -> List[str]:
    """변수 중요도 해석 문단."""
    imp = result.feature_importance
    if imp is None or imp.empty:
        return ["변수 중요도를 산출하지 못했습니다."]

    top = imp.head(top_n)
    names = ", ".join(f"'{r['변수']}'({r['중요도(%)']:.1f}%)" for _, r in top.iterrows())
    out = [f"예측에 가장 크게 기여한 상위 {len(top)}개 변수는 {names} 순입니다."]

    share = float(top["중요도(%)"].sum())
    if share >= 80:
        out.append(
            f"상위 {len(top)}개 변수가 전체 중요도의 {share:.0f}%를 차지하여, "
            "소수 변수에 예측력이 집중되어 있습니다. 해당 변수의 데이터 품질 관리가 중요합니다."
        )
    elif share <= 40:
        out.append(
            f"상위 {len(top)}개 변수의 합이 {share:.0f}%에 그쳐 여러 변수가 고르게 기여하는 구조입니다."
        )

    negligible = int((imp["중요도(%)"] < 0.5).sum())
    if negligible >= 3:
        out.append(
            f"기여도가 0.5% 미만인 변수가 {negligible}개 있어, 이를 제외하면 더 단순하고 "
            "유지보수하기 쉬운 모델을 만들 수 있습니다."
        )
    return out


def recommendation_sentences(
    profile: DataProfile, result: Optional[ModelingResult]
) -> List[str]:
    """실무 적용 권고 문단."""
    out: List[str] = []

    if profile.n_rows < 500:
        out.append(
            f"관측치가 {profile.n_rows:,}건으로 적어 성능 지표의 변동성이 큽니다. "
            "데이터를 추가 확보한 뒤 재학습하는 것을 권장합니다."
        )

    high_missing = [c for c in profile.columns.values() if c.missing_ratio > 0.3]
    if high_missing:
        out.append(
            f"결측률 30% 초과 컬럼이 {len(high_missing)}개 있습니다. "
            "수집 단계에서 누락 원인을 파악하면 모델 성능을 추가로 개선할 수 있습니다."
        )

    if result is not None:
        if result.task_type == "classification" and result.confusion is not None:
            cm = result.confusion
            if cm.shape[0] == cm.shape[1] and cm.shape[0] > 1:
                import numpy as np

                row_totals = cm.sum(axis=1).astype(float)
                denom = np.where(row_totals > 0, row_totals, np.nan)
                per_class = cm.diagonal() / denom
                if np.all(np.isnan(per_class)):
                    return out
                worst_idx = int(np.nanargmin(per_class))
                labels = result.class_labels or [str(i) for i in range(cm.shape[0])]
                if worst_idx < len(labels):
                    out.append(
                        f"클래스 '{labels[worst_idx]}'의 재현율이 {per_class[worst_idx]:.1%}로 가장 낮아, "
                        "해당 유형의 사례를 추가 확보하거나 가중치 조정을 검토하십시오."
                    )
        out.append(
            "본 보고서의 모델은 학습 시점의 데이터에 기반합니다. 실제 업무 적용 시에는 "
            "주기적으로 최신 데이터로 재학습하여 성능 저하 여부를 점검하시기 바랍니다."
        )
    else:
        out.append(
            "본 보고서는 탐색적 분석 단계까지 수행했습니다. 목표 변수를 지정하면 "
            "예측 모델 비교와 변수 중요도 분석을 이어서 수행할 수 있습니다."
        )

    out.append(
        "상관관계는 인과관계를 의미하지 않으므로, 도출된 주요 변수는 현업 지식과 대조하여 해석하십시오."
    )
    return out


def executive_summary(
    profile: DataProfile,
    source_name: str,
    result: Optional[ModelingResult],
) -> List[str]:
    """보고서 첫머리 요약(3~5문장)."""
    out = [
        f"'{source_name}' 데이터 {profile.n_rows:,}건 × {profile.n_cols:,}열을 대상으로 "
        "자동 전처리, 탐색적 분석, 예측 모델링을 수행했습니다."
    ]

    if result is None:
        out.append("목표 변수를 지정하지 않아 탐색적 데이터 분석 결과만 수록했습니다.")
        return out

    if result.task_type == "regression":
        out.append(
            f"'{result.target}' 예측을 위해 {result.best_model_name} 모델을 선정했으며, "
            f"검증 데이터 결정계수(R²)는 {result.holdout_metrics.get('R2', 0):.3f} 입니다."
        )
    else:
        out.append(
            f"'{result.target}' 분류를 위해 {result.best_model_name} 모델을 선정했으며, "
            f"검증 정확도는 {result.holdout_metrics.get('정확도', 0):.1%}, "
            f"가중 F1은 {result.holdout_metrics.get('F1(가중)', 0):.3f} 입니다."
        )

    imp = result.feature_importance
    if imp is not None and not imp.empty:
        top3 = ", ".join(f"'{n}'" for n in imp.head(3)["변수"].tolist())
        out.append(f"결과에 가장 큰 영향을 준 변수는 {top3} 입니다.")

    return out
