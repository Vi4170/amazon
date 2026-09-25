"""Blocking keys: hashed token-pair combinations within name, within address, and across the two.

Single tokens are too common in this data (a small generator vocabulary), so every key is an
unordered pair of skeleton tokens, prefixed with country + key type so countries never mix.
Address tokens used for pairing are restricted to each record's rarest ADDR_K tokens (by
target-side document frequency) to bound the key count per record.
"""
from __future__ import annotations

import polars as pl

ADDR_K = 5  # rarest address tokens per record used for pairing
NAME_K = 6  # rarest name tokens per record used for pairing


def token_df(rec: pl.DataFrame, col: str) -> pl.DataFrame:
    """Target-side (S2+S3) document frequency per (country, token)."""
    return (
        rec.filter(pl.col("src") != 1)
        .select("country", pl.col(col).alias("tok"))
        .explode("tok")
        .drop_nulls("tok")
        .group_by("country", "tok")
        .agg(pl.len().alias("df"))
    )


def rarest(rec: pl.DataFrame, col: str, df: pl.DataFrame, k: int) -> pl.DataFrame:
    """(idx, country, tok, df) keeping each record's k rarest tokens (unseen tokens count as df=0)."""
    t = rec.select("idx", "country", pl.col(col).alias("tok")).explode("tok").drop_nulls("tok")
    t = t.filter(pl.col("tok").str.len_chars() >= 2).join(df, on=["country", "tok"], how="left")
    t = t.with_columns(pl.col("df").fill_null(0))
    return t.sort("idx", "df", "tok").group_by("idx", maintain_order=True).head(k)


TYPES = {"n": 0, "a": 1, "nn": 2, "aa": 3, "na": 4}


def _singles(a: pl.DataFrame, typ: str) -> pl.DataFrame:
    key = pl.concat_str([pl.col("country"), pl.lit(typ), pl.col("tok")], separator="|").hash()
    return a.select("idx", key.alias("key"), pl.lit(TYPES[typ], pl.Int8).alias("typ"))


def _pairs(a: pl.DataFrame, b: pl.DataFrame, typ: str, same: bool) -> pl.DataFrame:
    j = a.join(b, on=["idx", "country"], suffix="_b")
    j = j.filter(pl.col("tok") < pl.col("tok_b")) if same else j
    lo = pl.min_horizontal("tok", "tok_b") if same else pl.col("tok")
    hi = pl.max_horizontal("tok", "tok_b") if same else pl.col("tok_b")
    key = pl.concat_str([pl.col("country"), pl.lit(typ), lo, hi], separator="|").hash()
    return j.select("idx", key.alias("key"), pl.lit(TYPES[typ], pl.Int8).alias("typ"))


def keys(rec: pl.DataFrame, ndf: pl.DataFrame, adf: pl.DataFrame) -> pl.DataFrame:
    """All blocking keys (idx, key:u64, typ) for the given records."""
    n = rarest(rec, "ntok", ndf, NAME_K).select("idx", "country", "tok")
    a = rarest(rec, "atok", adf, ADDR_K).select("idx", "country", "tok")
    a3 = a.group_by("idx", maintain_order=True).head(3)
    n2 = n.group_by("idx", maintain_order=True).head(2)
    out = pl.concat([_singles(n, "n"), _singles(a, "a"), _pairs(n, n, "nn", True), _pairs(a, a, "aa", True),
                     _pairs(n2, a3, "na", False)])
    return out.unique(["idx", "key"])
