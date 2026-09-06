#!/usr/bin/env bash
# ===================================================================
#  Standalone Auto-EDA & Report Builder 실행 스크립트 (macOS / Linux)
# ===================================================================
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "[1/3] 가상환경을 생성합니다..."
  python3 -m venv .venv
  echo "[2/3] 필요한 패키지를 설치합니다. 수 분이 걸릴 수 있습니다..."
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
else
  echo "[1/3] 기존 가상환경을 사용합니다."
  echo "[2/3] 패키지 확인 완료."
fi

echo "[3/3] 브라우저에서 분석 도구를 실행합니다..."
.venv/bin/python -m streamlit run app.py
