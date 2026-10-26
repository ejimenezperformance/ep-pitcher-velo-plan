"""Download per-game fastball velocity for every MLB pitcher (regular season).

Source: Baseball Savant Statcast Search, grouped by pitcher and game date.
Two pulls per month: four-seam (FF) and sinker (SI), so each pitcher's
primary fastball can be tracked on its own (mixing FF and SI would move the
average when the pitch mix changes, not when the arm changes).

Output: data/raw/fastballs_<year>.csv  (one row per pitcher-game-pitch_type)
Usage:  python src/pull_data.py 2017 2026
"""
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://baseballsavant.mlb.com/statcast_search/csv"
KEEP = ["player_id", "player_name", "game_date", "game_pk", "pitches",
        "total_pitches", "velocity"]
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# (start, end) month windows; regular season only (hfGT=R)
WINDOWS = [("03-01", "03-31"), ("04-01", "04-30"), ("05-01", "05-31"),
           ("06-01", "06-30"), ("07-01", "07-31"), ("08-01", "08-31"),
           ("09-01", "09-30"), ("10-01", "10-15")]


def pull(year: int, pitch: str, start: str, end: str) -> pd.DataFrame:
    params = {
        "hfPT": f"{pitch}|", "hfGT": "R|", "hfSea": f"{year}|",
        "player_type": "pitcher", "group_by": "name-date",
        "game_date_gt": f"{year}-{start}", "game_date_lt": f"{year}-{end}",
        "min_pitches": 0, "min_results": 0, "sort_col": "pitches",
        "sort_order": "desc", "chk_stats_velocity": "on",
    }
    for attempt in range(4):
        r = requests.get(URL, params=params, timeout=120)
        if r.ok:
            break
        time.sleep(5 * (attempt + 1))
    r.raise_for_status()
    text = r.content.decode("utf-8-sig").strip()
    if not text:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text))
    if df.empty or "velocity" not in df.columns:
        return df
    if len(df) >= 24_999:
        raise RuntimeError(f"Row cap hit: {year} {pitch} {start}-{end}")
    df = df[KEEP].copy()
    df["pitch_type"] = pitch
    return df


AGE_URL = "https://baseballsavant.mlb.com/leaderboard/custom"


def pull_ages(year: int) -> pd.DataFrame:
    """Season age (Savant convention: age on June 30) for every pitcher."""
    params = {"year": year, "type": "pitcher", "filter": "", "min": 1,
              "selections": "player_age,pitch_count", "csv": "true"}
    r = requests.get(AGE_URL, params=params, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")))
    return df[["player_id", "year", "player_age"]]


def main(first: int, last: int) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    ages = RAW / "ages.csv"
    if not ages.exists():
        pd.concat([pull_ages(y) for y in range(first, last + 1)]).to_csv(
            ages, index=False, encoding="utf-8")
        print("ages saved")
    for year in range(first, last + 1):
        out = RAW / f"fastballs_{year}.csv"
        if out.exists():
            print("skip", out.name)
            continue
        parts = [pull(year, p, s, e) for p in ("FF", "SI") for s, e in WINDOWS]
        df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
        df.to_csv(out, index=False, encoding="utf-8")
        print(year, len(df), "rows")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
