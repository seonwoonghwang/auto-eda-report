# Streamlit Community Cloud 배포 가이드

이 문서 하나만 따라 하면 됩니다. 소요 시간은 처음이면 20~30분, 익숙해지면 5분입니다.

> **먼저 읽어 주세요 — 데이터 보안**
> Community Cloud는 **미국 서버**에서 앱을 실행합니다. 사용자가 올린 파일은 그 서버의
> 메모리와 임시 디스크에서 처리됩니다. 사내 매출·품질·인사 데이터를 여기서 다루는 것은
> 사내 정보보안 규정 검토 대상입니다.
> 외부 배포는 **샘플 데이터 시연이나 사외 공유용**으로 쓰시고, 실제 업무 데이터는
> 사내 PC/서버에서 실행하는 방식(README의 로컬 실행)을 권합니다.

---

## 0. 준비물

| 항목 | 확인 방법 |
|---|---|
| GitHub 계정 | https://github.com 가입 (무료) |
| Git 설치 | PowerShell에서 `git --version` 실행 시 버전이 나오면 OK |

Git이 없으면 https://git-scm.com/download/win 에서 설치하세요. 설치 중 옵션은 전부 기본값으로 두면 됩니다.

배포에 필요한 파일 세 개는 이미 프로젝트에 들어 있습니다.

- `requirements.txt` — 파이썬 패키지 목록
- `packages.txt` — 리눅스 패키지 목록. **한글 폰트(나눔고딕)를 여기서 설치합니다.** 이 파일이 없으면 차트의 한글이 전부 네모(□)로 나옵니다.
- `.streamlit/config.toml` — 테마와 업로드 용량 제한

---

## 1. GitHub 저장소 만들기

1. https://github.com/new 접속
2. **Repository name** 에 `auto-eda-report` 입력
3. **Public** 또는 **Private** 선택
   - Private으로 해도 배포됩니다. 다만 무료 계정은 **비공개 앱을 한 번에 하나만** 운영할 수 있습니다.
   - 코드에 회사 고유 정보가 없다면 Public이 관리하기 편합니다.
4. 나머지는 아무것도 체크하지 말고 (README, .gitignore, license 전부 해제) **Create repository** 클릭

빈 저장소가 만들어지고 안내 화면이 나옵니다. 그 화면은 닫아도 됩니다.

---

## 2. 코드 올리기

PowerShell을 열고 아래를 **한 줄씩** 실행하세요.
`<본인계정>` 부분만 실제 GitHub 아이디로 바꾸면 됩니다.

```powershell
cd C:\Users\snscu\projects\auto_eda_report

git init
git add .
git commit -m "데이터 분석 자동화 도구 최초 배포"
git branch -M main
git remote add origin https://github.com/<본인계정>/auto-eda-report.git
git push -u origin main
```

`git commit` 에서 이름/이메일을 물어보면 아래를 먼저 실행하고 다시 시도하세요.

```powershell
git config --global user.name "홍길동"
git config --global user.email "본인@이메일.com"
```

`git push` 할 때 로그인 창이 뜹니다. **Sign in with your browser** 를 눌러 GitHub에 로그인하면 됩니다.

올라간 파일 확인 — GitHub 저장소 페이지를 새로고침했을 때 `app.py`, `autoeda/`, `requirements.txt`, `packages.txt`, `samples/` 가 보이면 성공입니다.

> `outputs/` 폴더 안의 보고서 파일과 `.venv` 폴더는 `.gitignore`에 걸려 자동으로 제외됩니다. 일부러 빼실 필요 없습니다.

---

## 3. Community Cloud에 배포

1. https://share.streamlit.io 접속 → **Continue with GitHub** 으로 로그인
2. GitHub 권한 승인 화면이 나오면 **Authorize** 클릭
   - 저장소를 Private으로 만들었다면, 이때 비공개 저장소 접근 권한도 함께 허용해야 합니다.
3. 오른쪽 위 **Create app** → **Deploy a public app from GitHub** 선택
4. 아래처럼 입력합니다.

   | 항목 | 값 |
   |---|---|
   | Repository | `<본인계정>/auto-eda-report` |
   | Branch | `main` |
   | Main file path | `app.py` |
   | App URL | 원하는 주소 (예: `worldbridge-auto-eda`) |

5. **Advanced settings** 를 눌러 **Python version** 을 **3.11** 또는 **3.12** 로 지정합니다.
   기본값으로 두면 나중에 파이썬이 올라갈 때 동작이 달라질 수 있으니 명시하는 편이 안전합니다.
   Secrets 는 이 앱에서 쓰지 않으므로 비워 둡니다.
6. **Deploy** 클릭

빌드 로그가 흐르기 시작합니다. 패키지 설치 때문에 **첫 배포는 3~7분** 걸립니다.
화면이 멈춘 것처럼 보여도 정상이니 기다리세요.

---

## 4. 배포 후 반드시 확인할 것

앱이 뜨면 `샘플 데이터로 체험` → 목표 변수 `품질판정` 선택 → `분석 시작` 을 눌러 끝까지 돌려 보세요.

- [ ] **차트의 한글이 정상인가** — 네모(□)로 나오면 `packages.txt` 가 저장소에 올라가지 않은 것입니다. 파일을 추가하고 다시 푸시한 뒤, Cloud 화면 우측 하단 메뉴에서 **Reboot app** 을 누르세요.
- [ ] **Word 보고서가 받아지는가** — 다운로드 버튼으로 `.docx` 를 받아 열어 봅니다.
- [ ] **보고서 안 한글 폰트** — 문서는 맑은 고딕으로 지정되어 있어, 여는 PC에 맑은 고딕이 있으면(윈도우는 기본 설치) 정상입니다.

---

## 5. 공개 범위 설정

앱 화면 우측 하단 **⋮ → Settings → Sharing** 에서 바꿉니다.

- **Public** — 주소를 아는 사람은 누구나 접속
- **Private** — 초대한 이메일 주소만 접속. 초대받은 사람은 Google 계정이나 이메일 링크로 로그인합니다.

사내 인원만 쓰게 하려면 Private으로 두고 동료 이메일을 추가하세요. 무료 계정은 비공개 앱 하나까지입니다.

---

## 6. 수정 사항 반영

코드를 고친 뒤 아래 세 줄이면 자동으로 재배포됩니다. 별도 버튼을 누를 필요가 없습니다.

```powershell
cd C:\Users\snscu\projects\auto_eda_report
git add .
git commit -m "수정 내용 요약"
git push
```

반영까지 1~2분 걸립니다.

---

## 7. 자원 한계와 대처

무료 계정의 앱 한 대당 상한입니다.

| 항목 | 상한 |
|---|---|
| 메모리 | 2.7GB |
| CPU | 2코어 |
| 저장 공간 | 50GB |

`Argh. This app has gone over its resource limits` 라는 메시지가 뜨면 메모리를 초과한 것입니다. 다음 순서로 줄이세요.

1. 사이드바 **모델링 → 비교할 알고리즘 수** 를 8 → 4 로 낮춥니다.
2. **교차검증 폴드 수** 를 5 → 3 으로 낮춥니다.
3. **변수 중요도 계산** 체크를 해제합니다. 순열 중요도가 가장 무거운 단계입니다.
4. 그래도 넘치면 업로드 데이터의 행 수를 줄이세요. 대략 **10만 행 × 30열** 부터 부담이 됩니다.

`requirements.txt` 에서 XGBoost·LightGBM을 주석 처리해 둔 것도 같은 이유입니다. 필요하면 주석을 풀되, 메모리 여유를 먼저 확인하세요.

앱은 일정 시간 아무도 접속하지 않으면 잠자기 상태가 됩니다. 다시 접속하면 깨어나는데 30초쯤 걸립니다. 데이터가 사라지는 것은 아니지만, 생성해 둔 보고서 파일은 임시 저장이라 없어집니다. **보고서는 그때그때 다운로드해 두세요.**

---

## 8. 자주 겪는 오류

**`ModuleNotFoundError: No module named 'autoeda'`**
Main file path를 잘못 지정한 경우입니다. `app.py` 여야 하며, 하위 폴더 경로를 적으면 안 됩니다.

**빌드 로그에서 `Unable to locate package` 로 멈춤**
`packages.txt` 에 오타가 있거나 빈 줄에 공백이 들어간 경우입니다. 파일에는 `fonts-nanum` 처럼 패키지 이름만 한 줄씩 적혀 있어야 합니다.

**`installer returned a non-zero exit code`**
패키지 설치 실패입니다. Advanced settings의 Python 버전을 3.11로 바꾸고 Reboot 해 보세요.

**샘플 데이터 버튼이 안 보임**
`samples/*.csv` 가 저장소에 올라가지 않은 경우입니다. `git status` 로 확인하고 다시 푸시하세요.

**앱은 뜨는데 분석 시작 후 멈춤**
대부분 메모리 초과입니다. 7번 항목의 순서대로 설정을 낮추세요.

---

## 참고 문서

- [앱 배포하기 — Streamlit Docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)
- [의존성 관리 (requirements.txt / packages.txt)](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [앱 공유 및 공개 범위](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app)
- [앱 자원 한계](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
