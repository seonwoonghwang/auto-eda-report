"""데모용 샘플 데이터 생성 스크립트.

실행하면 같은 폴더에 회귀용/분류용 CSV 두 개를 만든다.

    python samples/make_samples.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
HERE = Path(__file__).resolve().parent


def make_sales(n: int = 900) -> pd.DataFrame:
    """월별 지점 매출 데이터(회귀 예측용)."""
    rng = np.random.default_rng(SEED)

    regions = ["수도권", "충청", "영남", "호남", "강원"]
    channels = ["직영점", "대리점", "온라인"]

    df = pd.DataFrame(
        {
            "일자": pd.to_datetime("2024-01-01")
            + pd.to_timedelta(rng.integers(0, 640, n), unit="D"),
            "지점코드": [f"BR{rng.integers(1, 46):03d}" for _ in range(n)],
            "지역": rng.choice(regions, n, p=[0.38, 0.15, 0.22, 0.17, 0.08]),
            "판매채널": rng.choice(channels, n, p=[0.4, 0.35, 0.25]),
            "영업사원수": rng.integers(2, 20, n),
            "광고비": rng.gamma(shape=2.2, scale=1_400_000, size=n).round(-3),
            "방문고객수": rng.poisson(320, n),
            "평균할인율": np.clip(rng.normal(0.11, 0.045, n), 0, 0.4).round(3),
            "고객만족도": np.clip(rng.normal(4.1, 0.5, n), 1, 5).round(2),
        }
    )

    region_effect = df["지역"].map(
        {"수도권": 1.32, "충청": 0.95, "영남": 1.08, "호남": 0.92, "강원": 0.78}
    )
    channel_effect = df["판매채널"].map({"직영점": 1.15, "대리점": 1.0, "온라인": 0.88})

    base = (
        9_500_000
        + df["광고비"] * 1.65
        + df["방문고객수"] * 21_000
        + df["영업사원수"] * 640_000
        + df["고객만족도"] * 1_900_000
        - df["평균할인율"] * 28_000_000
    )
    noise = rng.normal(0, 4_200_000, n)
    df["매출액"] = ((base * region_effect * channel_effect) + noise).round(-3)
    df["매출액"] = df["매출액"].clip(lower=1_000_000)

    # 현실감을 위한 결측치 주입
    for col, ratio in (("고객만족도", 0.06), ("평균할인율", 0.03), ("광고비", 0.02)):
        idx = rng.choice(n, int(n * ratio), replace=False)
        df.loc[idx, col] = np.nan

    return df


def make_quality(n: int = 1100) -> pd.DataFrame:
    """제조 공정 품질 판정 데이터(분류 예측용)."""
    rng = np.random.default_rng(SEED + 7)

    df = pd.DataFrame(
        {
            "로트번호": [f"LOT{i:05d}" for i in range(n)],
            "설비": rng.choice(["1호기", "2호기", "3호기", "4호기"], n, p=[0.3, 0.3, 0.25, 0.15]),
            "작업조": rng.choice(["A조", "B조", "C조"], n),
            "원자재공급사": rng.choice(["가공업체", "나공업체", "다공업체"], n, p=[0.5, 0.3, 0.2]),
            "용접전류": rng.normal(185, 14, n).round(1),
            "용접속도": rng.normal(42, 5.5, n).round(2),
            "예열온도": rng.normal(148, 18, n).round(1),
            "판두께": np.clip(rng.normal(6.2, 0.9, n), 3, 12).round(2),
            "작업시간": np.clip(rng.normal(37, 9, n), 8, 90).round(1),
            "습도": np.clip(rng.normal(56, 12, n), 20, 95).round(1),
        }
    )

    equipment_risk = df["설비"].map({"1호기": -0.4, "2호기": -0.2, "3호기": 0.55, "4호기": 0.9})
    supplier_risk = df["원자재공급사"].map(
        {"가공업체": -0.3, "나공업체": 0.15, "다공업체": 0.8}
    )

    logit = (
        -2.1
        + equipment_risk
        + supplier_risk
        + 0.055 * (df["용접전류"] - 185).abs()
        + 0.075 * (df["용접속도"] - 42).abs()
        - 0.020 * (df["예열온도"] - 148)
        + 0.019 * (df["습도"] - 56)
        + rng.normal(0, 0.45, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    df["품질판정"] = np.where(rng.random(n) < prob, "불량", "양품")

    idx = rng.choice(n, int(n * 0.04), replace=False)
    df.loc[idx, "예열온도"] = np.nan

    return df


def main() -> None:
    sales = make_sales()
    quality = make_quality()

    sales_path = HERE / "sample_sales_regression.csv"
    quality_path = HERE / "sample_quality_classification.csv"

    sales.to_csv(sales_path, index=False, encoding="utf-8-sig")
    quality.to_csv(quality_path, index=False, encoding="utf-8-sig")

    print(f"생성 완료: {sales_path.name} ({len(sales):,}행)")
    print(f"생성 완료: {quality_path.name} ({len(quality):,}행)")
    print(f"불량률: {(quality['품질판정'] == '불량').mean():.1%}")


if __name__ == "__main__":
    main()
