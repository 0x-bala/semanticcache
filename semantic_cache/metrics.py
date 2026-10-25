def cache_hit_rate(hits,misses):
    t=hits+misses
    return hits/t if t else 0.0
