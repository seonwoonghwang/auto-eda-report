"""로컬 데이터 파일(CSV/Excel) 적재 모듈.

한글 CSV에서 흔히 발생하는 인코딩(cp949/euc-kr) 문제와 구분자 문제를
자동으로 해결하는 것을 목표로 한다.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

import pandas as pd

from .config import ENCODING_CANDIDATES, SEPARATOR_CANDIDATES

SUPPORTED_SUFFIXES = {".csv", ".txt", ".tsv", ".xlsx", ".xls", ".xlsm"}
EXCEL_SUFFIXES = {".xlsx", ".xls", ".xlsm"}


class DataLoadError(RuntimeError):
    """데이터 적재 실패."""


@dataclass
class LoadResult:
    """적재 결과와 적재 과정에서 확인된 메타 정보."""

    df: pd.DataFrame
    source_name: str
    encoding: Optional[str] = None
    separator: Optional[str] = None
    sheet_name: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @property
    def shape_text(self) -> str:
        return f"{self.df.shape[0]:,}행 × {self.df.shape[1]:,}열"

    def meta_dict(self) -> Dict[str, Any]:
        meta = {"파일명": self.source_name, "데이터 크기": self.shape_text}
        if self.encoding:
            meta["인코딩"] = self.encoding
        if self.separator:
            meta["구분자"] = repr(self.separator)
        if self.sheet_name:
            meta["시트"] = self.sheet_name
        return meta


def _read_csv_bytes(raw: bytes) -> tuple[pd.DataFrame, str, str]:
    """바이트 스트림에서 인코딩·구분자를 탐색하며 CSV를 읽는다."""
    last_error: Optional[Exception] = None
    for encoding in ENCODING_CANDIDATES:
        for sep in SEPARATOR_CANDIDATES:
            try:
                kwargs: Dict[str, Any] = {"encoding": encoding}
                if sep is None:
                    kwargs.update({"sep": None, "engine": "python"})
                else:
                    kwargs["sep"] = sep
                df = pd.read_csv(io.BytesIO(raw), **kwargs)
            except Exception as exc:  # noqa: BLE001 - 후보 탐색 중 실패는 정상 흐름
                last_error = exc
                continue

            # 컬럼이 1개뿐이면 구분자를 잘못 잡은 것으로 보고 다음 후보 시도
            if df.shape[1] <= 1 and sep is not None:
                last_error = DataLoadError("구분자 추정 실패")
                continue
            if df.shape[1] == 0:
                continue
            return df, encoding, sep or "자동 추론"

    raise DataLoadError(f"CSV 파일을 읽지 못했습니다. 마지막 오류: {last_error}")


def list_excel_sheets(source: Union[str, Path, bytes, BinaryIO]) -> List[str]:
    """Excel 파일의 시트 목록을 반환한다."""
    buf = io.BytesIO(source) if isinstance(source, bytes) else source
    with pd.ExcelFile(buf) as xls:
        return list(xls.sheet_names)


def load_dataframe(
    source: Union[str, Path, bytes, BinaryIO],
    *,
    filename: Optional[str] = None,
    sheet_name: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> LoadResult:
    """CSV/Excel 파일을 DataFrame으로 적재한다.

    Parameters
    ----------
    source:
        파일 경로, 바이트, 또는 파일 객체(Streamlit UploadedFile 포함).
    filename:
        source 가 바이트/스트림일 때 확장자 판별에 사용할 원본 파일명.
    sheet_name:
        Excel 인 경우 읽을 시트명. None 이면 첫 번째 시트.
    max_rows:
        미리보기 등 대용량 데이터를 부분만 읽고 싶을 때 상한 행 수.
    """
    notes: List[str] = []

    # 1) 입력 정규화 -----------------------------------------------------
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DataLoadError(f"파일을 찾을 수 없습니다: {path}")
        name = path.name
        raw = path.read_bytes()
    else:
        name = filename or getattr(source, "name", "uploaded_data")
        raw = source if isinstance(source, bytes) else source.read()

    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DataLoadError(
            f"지원하지 않는 형식입니다: {suffix or '확장자 없음'} "
            f"(지원 형식: {', '.join(sorted(SUPPORTED_SUFFIXES))})"
        )

    # 2) 적재 ------------------------------------------------------------
    if suffix in EXCEL_SUFFIXES:
        sheets = list_excel_sheets(raw)
        target_sheet = sheet_name or (sheets[0] if sheets else 0)
        df = pd.read_excel(io.BytesIO(raw), sheet_name=target_sheet)
        result = LoadResult(df=df, source_name=name, sheet_name=str(target_sheet))
        if len(sheets) > 1:
            notes.append(f"시트 {len(sheets)}개 중 '{target_sheet}' 시트를 사용했습니다.")
    else:
        df, encoding, sep = _read_csv_bytes(raw)
        result = LoadResult(df=df, source_name=name, encoding=encoding, separator=sep)

    # 3) 후처리 ----------------------------------------------------------
    df = result.df
    original_shape = df.shape

    # 완전히 비어 있는 행/열 제거
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # 컬럼명 정리(공백 제거, 중복 컬럼 접미사 부여)
    df.columns = [str(c).strip() for c in df.columns]
    df = _deduplicate_columns(df)

    # 'Unnamed: 0' 형태의 인덱스 잔재 컬럼 제거
    junk = [
        c
        for c in df.columns
        if c.lower().startswith("unnamed:") and df[c].isna().mean() > 0.9
    ]
    if junk:
        df = df.drop(columns=junk)
        notes.append(f"빈 컬럼 {len(junk)}개를 제거했습니다: {', '.join(junk)}")

    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
        notes.append(f"상위 {max_rows:,}행만 사용합니다.")

    if df.shape != original_shape:
        notes.append(
            f"정리 전 {original_shape[0]:,}행×{original_shape[1]:,}열 → "
            f"정리 후 {df.shape[0]:,}행×{df.shape[1]:,}열"
        )

    if df.empty:
        raise DataLoadError("데이터가 비어 있습니다. 파일 내용을 확인해 주세요.")

    df = df.reset_index(drop=True)
    result.df = df
    result.notes.extend(notes)
    return result


def _deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """중복된 컬럼명에 _1, _2 접미사를 붙여 유일하게 만든다."""
    seen: Dict[str, int] = {}
    new_cols: List[str] = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    return df
