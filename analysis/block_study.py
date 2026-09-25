"""Blocking key study on a train S1 sample (full target pool): recall vs cost per df cap, and
recall@K after ranking candidates by IDF-weighted shared-key score.

Results (30k S1, run 1): cap20 39 pairs/S1 rec .927 | cap50 104 .956 | cap100 194 .967 |
cap300 528 .978 | cap1000 1519 .985.  By type @cap300: nn .66 aa .906 na .837 n .138 a .338
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code", "business_entity_resolution", "src"))
import guard  # noqa: F401,E402  (caps threads + RSS before polars loads)
import time  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import io_utils as u  # noqa: E402
import keys as K  # noqa: E402
import prep  # noqa: E402

N_S1 = int(os.environ.get("N_S1", "20000"))
CAPS = [int(c) for c in os.environ.get("CAPS", "300,1000").split(",")]
sys.stdout.reconfigure(encoding="utf-8")

rec = prep.records("train").select("idx", "entity_id", "src", "country", "ntok", "atok")
ndf, adf = K.token_df(rec, "ntok"), K.token_df(rec, "atok")
s1 = rec.filter(pl.col("src") == 1).sample(N_S1, seed=0)
t = time.time()
s1k = K.keys(s1, ndf, adf)
keyset = s1k.select("key").unique()
tg = rec.filter(pl.col("src") != 1).select("idx", "country", "ntok", "atok", "src")
Nt = tg.height
del rec
tk = pl.concat([K.keys(tg.slice(i, 500_000), ndf, adf).join(keyset, on="key", how="semi")
                for i in range(0, Nt, 500_000)])
print(f"target keys {tk.height:,}  {time.time() - t:.0f}s  peak {guard.PEAK[0]:.1f}GB", flush=True)
dft = tk.group_by("key").agg(pl.len().alias("dfT"))

gt = u.load("train", "ground_truth")
id2 = prep.records("train").select("entity_id", "idx")
gp = (gt.join(id2, left_on="source1_entity_id", right_on="entity_id")
      .join(s1.select("idx"), on="idx", how="semi")
      .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
      .filter(pl.col("matched_entity_ids") != "")
      .join(id2, left_on="matched_entity_ids", right_on="entity_id", suffix="_t")
      .select(pl.col("idx").alias("s"), pl.col("idx_t").alias("t")))
print("true pairs", gp.height)

sk = s1k.join(dft, on="key").rename({"idx": "s"})
for cap in CAPS:
    ks = sk.filter(pl.col("dfT") <= cap).with_columns((np.log(Nt) - pl.col("dfT").log()).alias("w"))
    cand = ks.join(tk.rename({"idx": "t"}), on=["key", "typ"])
    agg = cand.group_by("s", "t").agg(pl.col("w").sum().alias("sc"))
    agg = agg.with_columns(pl.col("sc").rank("ordinal", descending=True).over("s").alias("rk"))
    j = gp.join(agg, on=["s", "t"], how="left")
    print(f"cap {cap}: pairs/S1 {agg.height / N_S1:.0f} | recall@K " +
          " ".join(f"K{k}:{(j['rk'] <= k).sum() / gp.height:.4f}" for k in [10, 20, 30, 50, 80, 120, 200]) +
          f" | all:{j['rk'].is_not_null().mean():.4f}  peak {guard.PEAK[0]:.1f}GB", flush=True)
    miss = j.filter(pl.col("rk").is_null() | (pl.col("rk") > 50)).select("s", "t")

full = prep.records("train").select("idx", "business_name", "business_address")
m = miss.sample(min(25, miss.height), seed=1).join(full, left_on="s", right_on="idx").join(
    full, left_on="t", right_on="idx", suffix="_t")
for r in m.select("business_name", "business_address", "business_name_t", "business_address_t").rows():
    print(" | ".join(r))
