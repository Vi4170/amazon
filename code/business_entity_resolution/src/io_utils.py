"""Loading raw challenge TSVs (cached as parquet) and writing submission TSVs."""
from __future__ import annotations

import csv
import os
from pathlib import Path

import polars as pl

COLS = ["entity_id", "business_name", "business_address", "country"]


def data_dir() -> Path:
    return Path(os.environ.get("ER_DATA", Path(__file__).resolve().parents[3] / "student_resource" / "dataset"))


def work_dir() -> Path:
    p = Path(os.environ.get("ER_WORK", Path(__file__).resolve().parents[3] / "work"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_tsv(path: Path) -> pl.DataFrame:
    # quote_char=None: names/addresses may contain raw quotes; everything is a string, empty -> "".
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                     missing_utf8_is_empty_string=True)
    return df.with_columns(pl.all().fill_null(""))


def load(split: str, name: str) -> pl.DataFrame:
    """split in {train,test}; name in {source1,source2,source3,ground_truth}. Parquet-cached."""
    cache = work_dir() / f"{split}_{name}.parquet"
    if not cache.exists():
        read_tsv(data_dir() / split / f"{split}_{name}.tsv").write_parquet(cache)
    return pl.read_parquet(cache)


def write_id_lists(path: Path, s1_ids: list[str], lists: dict[str, list[str]], col: str) -> None:
    """One row per S1 id (in given order), comma-joined list, empty for none. Plain UTF-8 TSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s1 in s1_ids:
            f.write(f"{s1}\t{','.join(dict.fromkeys(lists.get(s1, ())))}\n")
