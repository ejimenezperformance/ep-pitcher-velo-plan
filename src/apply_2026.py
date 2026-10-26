"""Step 2: apply the baseline to 2026. Every pitcher whose primary fastball
fell >= 1 mph from 2025 to 2026 gets a lane:
  MEDICAL FIRST  - an absence (>=14-day gap, late start or early end) or a
                   known injury in data/injury_exclusions.csv: clear with
                   medical staff before any velocity work.
  ROLE CHANGE    - moved from bullpen to rotation (velocity usually drops).
  PHYSICAL PLAN  - no alternative explanation found: 12-week offseason plan.
"""
from pathlib import Path
import pandas as pd
import velo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
panel = pd.read_csv(OUT / "season_panel.csv")
e = velo.drop_events(panel, [2026])
d = e[e["is_drop"]].copy()
inj_path = ROOT / "data" / "injury_exclusions.csv"
inj = pd.read_csv(inj_path) if inj_path.exists() else pd.DataFrame(columns=["player_name", "action", "injury_type", "reason"])
d = d.merge(inj, on="player_name", how="left")

def lane(r):
    if r.get("action") == "exclude" or r.absence == 1:
        return "MEDICAL FIRST"
    if r.role_to_sp == 1:
        return "ROLE CHANGE"
    return "PHYSICAL PLAN"
d["lane"] = d.apply(lane, axis=1)
cols = ["player_name", "pitch_type", "age", "velo_base", "velo_drop_year", "drop_mph",
        "velo_first5", "fade_mph", "early_down", "absence", "max_gap", "role_to_sp", "role_to_rp",
        "fb_drop_year", "injury_type", "reason", "lane"]
first5 = panel[panel.year == 2026].set_index("player_id")["velo_first5"]
d["velo_first5"] = d["player_id"].map(first5)
d = d.sort_values(["lane", "drop_mph"])[cols].round(2)
d.to_csv(OUT / "board_2026.csv", index=False, encoding="utf-8")
print(d.to_string())
print(d["lane"].value_counts())
