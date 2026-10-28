import json
import sys

import numpy as np
from sentence_transformers import CrossEncoder

sys.path.insert(0, ".")
from semantic_cache.guard import GuardVerdict, check_match

pairs = json.load(open("benchmarks/data/pairs_2k.json", encoding="utf-8"))
verify = [p for p in pairs
          if check_match(p["query"], p["cached_q"]).verdict == GuardVerdict.VERIFY]
print(f"VERIFY pairs: {len(verify)} (duplicates: {sum(p['is_duplicate'] for p in verify)})")

model = CrossEncoder("cross-encoder/stsb-roberta-base", max_length=256)
scores = np.asarray(model.predict(
    [[p["query"], p["cached_q"]] for p in verify],
    batch_size=32, show_progress_bar=True)).reshape(-1)
y = np.array([p["is_duplicate"] for p in verify])

print("duplicate scores p10/p50/p90:    ", np.percentile(scores[y == 1], [10, 50, 90]).round(2))
print("non-duplicate scores p10/p50/p90:", np.percentile(scores[y == 0], [10, 50, 90]).round(2))
print(f"{'thr':>5} {'dups recovered':>15} {'non-dups accepted':>18}")
for t in [2.5, 3.0, 3.5, 3.75, 4.0, 4.25, 4.5]:
    acc = scores >= t
    print(f"{t:>5} {acc[y == 1].mean():>15.1%} {acc[y == 0].mean():>18.1%}")