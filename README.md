# Standalone Auto-EDA & Report Builder

외부 LLM API에 전혀 의존하지 않는 **독립형(Standalone) 데이터 분석 자동화 도구**입니다.
CSV·Excel 파일을 올리고 목표 변수만 고르면 전처리 → 통계 분석 → 예측 모델링 → **Word 보고서**까지
한 번에 만들어 줍니다. 코딩 지식은 필요하지 않습니다.

- 인터넷 연결 없이 로컬에서만 동작하며, 데이터가 외부로 전송되지 않습니다.
- 화면과 보고서 모두 한국어이며, 한글 폰트(맑은 고딕)를 자동 적용합니다.

---

## 1. 빠른 시작

### Windows

`run_app.bat` 을 더블클릭하면 가상환경 생성 → 패키지 설치 → 브라우저 실행까지 자동으로 진행됩니다.
(최초 1회는 패키지 설치로 수 분이 걸립니다.)

### macOS / Linux

```bash
chmod +x run_app.sh
./run_app.sh
```

### 수동 설치

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
streamlit run app.py
```

브라우저가 자동으로 열리지 않으면 http://localhost:8501 로 접속하세요.

---

## 2. 사용 방법 (5단계)

| 단계 | 화면 | 하는 일 |
|---|---|---|
| STEP 1 | 데이터 업로드 | CSV·XLSX 파일을 끌어다 놓습니다. 한글 인코딩(cp949/euc-kr)과 구분자를 자동 인식합니다. |
| STEP 2 | 컬럼 타입 확인 | 수치형/범주형/날짜형 자동 인식 결과를 확인하고, 필요하면 드롭다운으로 수정합니다. |
| STEP 3 | 분석 설정 | 예측할 **목표 변수**를 고르고, 제외할 컬럼을 선택합니다. 회귀/분류는 자동 판정됩니다. |
| STEP 4 | 분석 실행 | 버튼 한 번으로 전처리 → 모델 비교 → 차트 생성 → 보고서 작성이 진행됩니다. |
| STEP 5 | 결과 확인 | 성능·주요 변수·차트를 화면에서 확인하고 `.docx` 보고서를 내려받습니다. |

목표 변수를 고르지 않으면 **탐색적 분석(EDA) 보고서**만 생성됩니다.

### 샘플 데이터로 체험하기

STEP 1의 `샘플 데이터로 체험` 버튼을 누르면 아래 데모 데이터가 즉시 로드됩니다.

| 파일 | 목표 변수 | 문제 유형 |
|---|---|---|
| `samples/sample_sales_regression.csv` | `매출액` | 회귀(수치 예측) |
| `samples/sample_quality_classification.csv` | `품질판정` | 분류(양품/불량 판별) |

샘플을 다시 만들려면 `python samples/make_samples.py` 를 실행하세요.

---

## 3. 명령줄(CLI) 사용

GUI 없이 배치로 돌릴 때 사용합니다.

```bash
python cli.py samples/sample_sales_regression.csv --target 매출액
python cli.py 데이터.xlsx --target 품질판정 --type classification --sheet Sheet1
python cli.py 데이터.csv --exclude 사번 성명 --out C:\보고서
python cli.py 데이터.csv                      # 목표 변수 없이 EDA 보고서만
```

주요 옵션: `--target` `--type` `--sheet` `--exclude` `--out` `--title` `--author` `--org`
`--cv` `--max-models` `--seed` `--quiet`

---

## 4. 파이썬 코드에서 직접 호출

```python
from autoeda import AnalysisConfig, run_from_file

config = AnalysisConfig(
    target="매출액",
    analysis_type="auto",          # auto | regression | classification | eda
    exclude_columns=["지점코드"],
    report_title="2026년 상반기 매출 분석",
)
result = run_from_file("sales.csv", config, output_dir="outputs")

print(result.task_type)                       # regression
print(result.modeling.best_model_name)        # 예: 히스토그램 부스팅
print(result.modeling.holdout_metrics)        # {'R2': 0.78, 'RMSE': ...}
print(result.report_path)                     # outputs/Analysis_Report_YYYYMMDD_HHMMSS.docx
```

---

## 5. 자동 처리 내용

### 5.1 데이터 적재
- 인코딩 자동 탐색: `utf-8-sig → utf-8 → cp949 → euc-kr → latin1`
- 구분자 자동 추론: 쉼표, 세미콜론, 탭, 파이프
- 빈 행·빈 열 제거, 중복 컬럼명 정리, Excel 다중 시트 선택

### 5.2 컬럼 타입 추론
- 수치형 / 범주형 / 날짜형 / 텍스트(식별자) 4종으로 분류
- 고유값이 20개 이하인 정수 컬럼은 코드성 범주형으로 판정
- `"1,234"` 같은 천단위 콤마 문자열은 수치형으로 복원
- 결측률 60% 초과·값이 한 종류뿐·식별자 성격 컬럼은 **제외 권고**로 표시

### 5.3 전처리 (모두 하나의 scikit-learn 파이프라인으로 재현)
- 결측치: 수치형은 중앙값/평균, 범주형은 최빈값 대체
- 이상치: IQR(1.5배) 경계로 윈저라이징
- 날짜: 연/월/일/요일 파생 변수 생성
- 인코딩: 고유값 30개 이하 One-hot, 초과 시 순서형 인코딩
- 스케일링: 표준화 또는 정규화(트리 계열 모델은 자동으로 생략)

### 5.4 AutoML
회귀·분류 문제를 자동 판정하고, 아래 알고리즘을 **동일한 교차검증 조건**으로 비교하여
가장 성능이 좋은 모델을 선택합니다.

| 회귀 | 분류 |
|---|---|
| 선형 회귀, 릿지, 엘라스틱넷 | 로지스틱 회귀 |
| 의사결정나무, 랜덤 포레스트, 엑스트라 트리 | 의사결정나무, 랜덤 포레스트, 엑스트라 트리 |
| 그래디언트 부스팅, 히스토그램 부스팅 | 그래디언트 부스팅, 히스토그램 부스팅 |
| K-최근접 이웃 | K-최근접 이웃 |
| *(선택) XGBoost, LightGBM* | *(선택) XGBoost, LightGBM* |

- 평가 지표 — 회귀: R², RMSE, MAE, MAPE / 분류: 정확도, F1, 정밀도, 재현율, ROC-AUC
- 최빈값·평균만 예측하는 기준 모델(Dummy) 대비 개선율을 함께 제시
- **변수 중요도**는 순열 중요도(permutation importance)를 원본 컬럼 단위로 계산하므로,
  One-hot으로 쪼개진 더미 변수가 아니라 실무자가 아는 **원래 컬럼 이름**으로 나옵니다.

> XGBoost·LightGBM은 선택 사항입니다. 설치되어 있으면 후보에 자동 포함되고, 없어도 정상 동작합니다.
> `pip install xgboost lightgbm`

### 5.5 Word 보고서 구성
표지 → 목차 → 1. 분석 요약 → 2. 데이터 개요 → 3. 자동 전처리 내역 → 4. 탐색적 데이터 분석 →
5. 예측 모델링 → 6. 주요 영향 변수 → 7. 결론 및 실무 권고사항 → 부록(분석 환경·설정)

- 통계표, 차트 이미지, 해석 문장이 정해진 위치에 자동 삽입됩니다.
- 해석 문장은 통계값 임계치에 따른 **규칙 기반 템플릿**으로 생성되며, 외부 AI를 사용하지 않습니다.
- 파일명: `Analysis_Report_YYYYMMDD_HHMMSS.docx`
- 목차 페이지 번호는 Word에서 목차를 클릭한 뒤 **F9** 를 누르면 갱신됩니다.

---

## 6. 프로젝트 구조

```
auto_eda_report/
├── app.py                  # Streamlit GUI (5단계 화면)
├── cli.py                  # 명령줄 진입점
├── run_app.bat             # Windows 실행 스크립트
├── run_app.sh              # macOS/Linux 실행 스크립트
├── requirements.txt
├── .streamlit/config.toml  # 화면 테마 및 업로드 용량 설정
│
├── autoeda/                # 분석 엔진 (GUI와 완전히 분리)
│   ├── config.py           # 설정·상수·한글 폰트·색상 팔레트
│   ├── data_loader.py      # CSV/Excel 적재, 인코딩·구분자 자동 인식
│   ├── profiling.py        # 컬럼 타입 추론, 기초 통계, 상관계수
│   ├── preprocessing.py    # 정제 + ColumnTransformer 파이프라인
│   ├── automl.py           # 모델 비교·선정·평가·변수 중요도
│   ├── visualization.py    # 차트 생성 (matplotlib)
│   ├── insights.py         # 규칙 기반 한국어 해석 문장 생성
│   ├── report_builder.py   # python-docx 보고서 조립
│   └── pipeline.py         # 전체 흐름 오케스트레이터
│
├── samples/                # 데모 데이터 및 생성 스크립트
├── tests/                  # 엔드투엔드 검증 스크립트
└── outputs/                # 생성된 보고서 저장 폴더(기본값)
```

GUI(`app.py`)와 엔진(`autoeda/`)이 분리되어 있어, 다른 화면이나 배치 시스템에
엔진만 그대로 가져다 쓸 수 있습니다.

---

## 7. 검증

```bash
python tests/test_end_to_end.py
```

컬럼 타입 추론, 문제 유형 판정, 회귀·분류·EDA 전체 파이프라인, 예외 상황(전부 결측/상수 컬럼/
소량 데이터), 입출력 형식(cp949·세미콜론·Excel), 보고서 구조까지 **42개 항목**을 점검합니다.

---

## 8. 자주 묻는 질문

**차트의 한글이 네모(□)로 나옵니다**
한글 폰트를 찾지 못한 경우입니다. Windows에는 맑은 고딕이 기본 설치되어 있으며,
Linux는 `sudo apt install fonts-nanum` 으로 나눔고딕을 설치하면 해결됩니다.

**얼마나 큰 파일까지 처리할 수 있나요**
기본 업로드 한도는 500MB(`.streamlit/config.toml`)이며, 실제 처리 속도는 PC 사양에 따라 다릅니다.
수십만 행 이상이면 사이드바에서 비교할 알고리즘 수와 교차검증 폴드를 줄이면 빨라집니다.

**모델 성능이 낮게 나옵니다**
① STEP 2에서 컬럼 타입이 맞는지 ② STEP 3에서 식별번호 같은 무의미한 컬럼을 제외했는지
③ 목표 변수를 설명할 수 있는 변수가 데이터에 실제로 들어 있는지 순서로 점검하세요.
보고서 7장에 상황별 권고사항이 자동으로 기재됩니다.

**보고서 양식을 회사 표준으로 바꾸고 싶습니다**
`autoeda/report_builder.py` 상단의 `ACCENT`, `HEADER_FILL`, `BODY_FONT` 값을 수정하면
색상과 폰트가 일괄 변경됩니다. 기존 `.docx` 템플릿을 쓰려면
`ReportBuilder(config, template=Path("사내양식.docx"))` 로 전달하세요.

---

## 9. 요구 사항

- Python 3.10 이상
- pandas, numpy, scikit-learn, matplotlib, python-docx, streamlit, openpyxl (`requirements.txt`)
- 선택: xgboost, lightgbm
# auto-eda-report
# auto-eda-report
