@echo off
chcp 65001 > nul
REM ===================================================================
REM  Standalone Auto-EDA & Report Builder 실행 스크립트 (Windows)
REM  최초 실행 시 가상환경 생성과 패키지 설치를 자동으로 수행합니다.
REM ===================================================================
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] 가상환경을 생성합니다...
    python -m venv .venv
    if errorlevel 1 (
        echo [오류] Python 을 찾을 수 없습니다. Python 3.10 이상을 설치한 뒤 다시 실행하세요.
        pause
        exit /b 1
    )
    echo [2/3] 필요한 패키지를 설치합니다. 수 분이 걸릴 수 있습니다...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
) else (
    echo [1/3] 기존 가상환경을 사용합니다.
    echo [2/3] 패키지 확인 완료.
)

echo [3/3] 브라우저에서 분석 도구를 실행합니다...
".venv\Scripts\python.exe" -m streamlit run app.py
pause
