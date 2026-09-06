"""명령줄 실행 진입점 (GUI 없이 배치 처리할 때 사용).

예시::

    python cli.py samples/sample_sales_regression.csv --target 매출액
    python cli.py data.xlsx --target 품질판정 --type classification --sheet Sheet1
    python cli.py data.csv                      # 목표 변수 없이 EDA 보고서만
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from autoeda.config import AnalysisConfig, DEFAULT_OUTPUT_DIR
from autoeda.data_loader import DataLoadError
from autoeda.pipeline import run_from_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-eda",
        description="CSV/Excel 데이터를 자동 분석하고 Word 보고서를 생성합니다.",
    )
    parser.add_argument("input", help="분석할 CSV 또는 Excel 파일 경로")
    parser.add_argument("--target", "-t", default=None, help="목표 변수(예측 대상) 컬럼명")
    parser.add_argument(
        "--type",
        dest="analysis_type",
        default="auto",
        choices=["auto", "regression", "classification", "eda"],
        help="분석 유형 (기본: auto)",
    )
    parser.add_argument("--sheet", default=None, help="Excel 시트명")
    parser.add_argument("--exclude", nargs="*", default=[], help="제외할 컬럼명 목록")
    parser.add_argument("--out", "-o", default=str(DEFAULT_OUTPUT_DIR), help="보고서 저장 폴더")
    parser.add_argument("--title", default="데이터 분석 자동화 보고서", help="보고서 제목")
    parser.add_argument("--author", default="", help="작성자")
    parser.add_argument("--org", default="", help="소속")
    parser.add_argument("--cv", type=int, default=5, help="교차검증 폴드 수 (기본: 5)")
    parser.add_argument("--max-models", type=int, default=8, help="비교할 알고리즘 수 (기본: 8)")
    parser.add_argument("--seed", type=int, default=42, help="난수 시드")
    parser.add_argument("--quiet", action="store_true", help="진행 상황 출력 생략")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = AnalysisConfig(
        analysis_type=args.analysis_type,
        target=args.target,
        exclude_columns=list(args.exclude),
        cv_folds=args.cv,
        max_models=args.max_models,
        random_state=args.seed,
        report_title=args.title,
        author=args.author,
        organization=args.org,
    )

    def progress(ratio: float, message: str) -> None:
        if not args.quiet:
            print(f"[{ratio * 100:3.0f}%] {message}", flush=True)

    try:
        result = run_from_file(
            Path(args.input),
            config,
            sheet_name=args.sheet,
            output_dir=Path(args.out),
            progress=progress,
        )
    except DataLoadError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"[오류] 분석에 실패했습니다: {exc}", file=sys.stderr)
        return 1

    print("\n" + "=" * 64)
    print(f"분석 유형 : {result.task_type}")
    if result.modeling:
        print(f"선정 모델 : {result.modeling.best_model_name}")
        for key, value in result.modeling.holdout_metrics.items():
            print(f"  - {key}: {value:,.4f}")
        if not result.modeling.feature_importance.empty:
            print("주요 변수 :")
            for _, row in result.modeling.feature_importance.head(5).iterrows():
                print(f"  - {row['변수']}: {row['중요도(%)']:.1f}%")
    print(f"보고서    : {result.report_path}")
    print("=" * 64)

    for message in result.messages:
        print(f"※ {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
