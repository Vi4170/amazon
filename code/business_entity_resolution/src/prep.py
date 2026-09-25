"""Stage 1: normalize all records of a split into one table (parquet-cached).

Columns: idx (int32 row id, S1 first then S2, S3), entity_id, src (1/2/3), country,
nf / af (folded name / address), ntok / atok (deduplicated phonetic-skeleton token lists).
"""
from __future__ import annotations

import polars as pl

import io_utils as u
from normalize import fold, skeleton


def _skel_lists(df: pl.DataFrame, col: str, out: str) -> pl.DataFrame:
    toks = df.select("idx", pl.col(col).str.split(" ").alias("t")).explode("t").filter(pl.col("t") != "")
    vocab = toks.select(pl.col("t").unique())
    vocab = vocab.with_columns(pl.col("t").map_elements(skeleton, return_dtype=pl.String).alias("s"))
    toks = toks.join(vocab, on="t", how="left").select("idx", "s").unique()
    lists = toks.group_by("idx").agg(pl.col("s").alias(out))
    return df.join(lists, on="idx", how="left").with_columns(pl.col(out).fill_null([]))


def records(split: str) -> pl.DataFrame:
    cache = u.work_dir() / f"{split}_records.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    parts = [u.load(split, f"source{k}").with_columns(pl.lit(k, pl.Int8).alias("src")) for k in (1, 2, 3)]
    df = pl.concat(parts).with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int32))
    df = df.with_columns(
        fold(pl.col("business_name")).alias("nf"),
        fold(pl.col("business_address")).alias("af"),
    )
    df = _skel_lists(df, "nf", "ntok")
    df = _skel_lists(df, "af", "atok").sort("idx")
    df.write_parquet(cache)
    return df


if __name__ == "__main__":
    import sys
    import time

    for sp in sys.argv[1:] or ["train", "test"]:
        t = time.time()
        d = records(sp)
        print(sp, d.shape, f"{time.time() - t:.0f}s")
