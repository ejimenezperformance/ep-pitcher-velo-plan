"""Robustness: does the low natural-recovery rate hold under other thresholds?"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
import velo

OUT = Path(__file__).resolve().parents[1] / "outputs"
panel = pd.read_csv(OUT / "season_panel.csv")
rows = []
for drop in (-1.0, -1.5, -2.0):
    for nxt in (200, 100):
        for share in (0.5, 0.25, 1.0):
            velo.DROP_MPH, velo.MIN_FB_NEXT, velo.RECOVER_SHARE = drop, nxt, share
            e = velo.drop_events(panel, velo.EVENT_YEARS)
            d = e[e["is_drop"]]
            h = d[d["status"] != "no_mlb_data"]
            lo, hi = proportion_confint(h["recovered"].sum(), len(h), method="wilson")
            rows.append({"drop_mph": drop, "min_fb_next": nxt, "recover_share": share,
                         "drops": len(d), "with_next_data": len(h),
                         "no_next_data_share": round(1 - len(h) / len(d), 3),
                         "recovery_rate": round(h["recovered"].mean(), 3),
                         "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
s = pd.DataFrame(rows)
s.to_csv(OUT / "sensitivity.csv", index=False, encoding="utf-8")
print(s.to_string())

# Why "no next-year data"? (default thresholds)
velo.DROP_MPH, velo.MIN_FB_NEXT, velo.RECOVER_SHARE = -1.0, 200, 0.5
e = velo.drop_events(panel, velo.EVENT_YEARS); d = e[e["is_drop"] & (e["status"] == "no_mlb_data")]
p = panel.set_index(["player_id", "year"])
def why(r):
    k = (r.player_id, r.year + 1)
    if k not in p.index: return "no MLB fastballs next year"
    n = p.loc[k]
    if n.pitch_type != r.pitch_type: return "changed primary fastball"
    return "under 200 fastballs next year"
print(d.apply(why, axis=1).value_counts())
