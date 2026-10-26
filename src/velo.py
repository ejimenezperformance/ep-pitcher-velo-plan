"""Core logic: season panel, velocity-drop events, recovery outcome, tests.

All functions are pure (DataFrame in, DataFrame out) so the synthetic tests in
tests/test_synthetic.py can run them on data with a known answer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ---- Thresholds (fixed before looking at results) ----
MIN_FB_BASE = 300        # fastballs in the base year and the drop year
MIN_FB_NEXT = 200        # fastballs in the outcome year
DROP_MPH = -1.0          # season velocity change that counts as a drop
RECOVER_SHARE = 0.5      # recovered = regained >= 50% of the drop
GAP_DAYS = 14            # appearance gap that flags an absence
STARTER_PITCHES = 50     # a game with >= 50 total pitches counts as a start
ALPHA = 0.05
MIN_EFFECT_PP = 10.0     # practical threshold: 10 percentage points
MIN_AUC = 0.60           # practical threshold for the model
EVENT_YEARS = [2018, 2022, 2023, 2024, 2025]  # no pair touches 2020

FEATURES = ["drop_mph", "age", "absence", "role_to_sp", "early_down",
            "fade_mph", "fb_base"]


def season_panel(games: pd.DataFrame, ages: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per pitcher-season, using that season's primary fastball."""
    g = games.copy()
    g["game_date"] = pd.to_datetime(g["game_date"])
    g["year"] = g["game_date"].dt.year
    g["vw"] = g["velocity"] * g["pitches"]

    by_type = (g.groupby(["player_id", "year", "pitch_type"])
                 .agg(n=("pitches", "sum")).reset_index())
    prim = by_type.sort_values("n", ascending=False).drop_duplicates(["player_id", "year"])
    g = g.merge(prim[["player_id", "year", "pitch_type"]], on=["player_id", "year", "pitch_type"])

    rows = []
    for (pid, yr), d in g.groupby(["player_id", "year"]):
        d = d.sort_values("game_date")
        n = d["pitches"].sum()
        velo = d["vw"].sum() / n
        first5 = d.head(5)
        v5 = first5["vw"].sum() / first5["pitches"].sum()
        h1 = d[d["game_date"].dt.month < 7]
        h2 = d[d["game_date"].dt.month >= 7]
        v_h1 = h1["vw"].sum() / h1["pitches"].sum() if h1["pitches"].sum() >= 50 else np.nan
        v_h2 = h2["vw"].sum() / h2["pitches"].sum() if h2["pitches"].sum() >= 50 else np.nan
        dates = d["game_date"].drop_duplicates()
        gaps = dates.diff().dt.days.dropna()
        start = pd.Timestamp(f"{yr}-05-01")
        end = pd.Timestamp(f"{yr}-09-01")
        rows.append({
            "player_id": pid, "player_name": d["player_name"].iloc[0], "year": yr,
            "pitch_type": d["pitch_type"].iloc[0], "fb": n, "velo": velo,
            "velo_first5": v5, "velo_h1": v_h1, "velo_h2": v_h2,
            "games": len(dates),
            "starter_share": (d["total_pitches"] >= STARTER_PITCHES).mean(),
            "max_gap": gaps.max() if len(gaps) else 0,
            "late_start": dates.min() > start, "early_end": dates.max() < end,
        })
    p = pd.DataFrame(rows)
    p["absence"] = ((p["max_gap"] >= GAP_DAYS) | p["late_start"] | p["early_end"]).astype(int)
    if ages is not None:
        ages = ages.drop_duplicates(["player_id", "year"])
        p = p.merge(ages, on=["player_id", "year"], how="left").rename(columns={"player_age": "age"})
    return p


def drop_events(panel: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Pitchers whose primary fastball fell >= 1 mph from year t-1 to t,
    with the outcome in t+1 (recovered / not recovered / no MLB data)."""
    p = panel.set_index(["player_id", "year"])
    out = []
    for t in years:
        cur = panel[panel["year"] == t]
        for _, r in cur.iterrows():
            key_prev, key_next = (r.player_id, t - 1), (r.player_id, t + 1)
            if key_prev not in p.index:
                continue
            prev = p.loc[key_prev]
            if prev.pitch_type != r.pitch_type or prev.fb < MIN_FB_BASE or r.fb < MIN_FB_BASE:
                continue
            delta = r.velo - prev.velo
            row = {
                "player_id": r.player_id, "player_name": r.player_name, "year": t,
                "pitch_type": r.pitch_type, "velo_base": prev.velo, "velo_drop_year": r.velo,
                "drop_mph": delta, "age": r.get("age", np.nan), "absence": r.absence,
                "max_gap": r.max_gap, "fb_base": prev.fb, "fb_drop_year": r.fb,
                "role_to_sp": int(prev.starter_share < 0.5 <= r.starter_share),
                "role_to_rp": int(prev.starter_share >= 0.5 > r.starter_share),
                "early_down": int(r.velo_first5 - prev.velo <= DROP_MPH),
                "fade_mph": (r.velo_h2 - r.velo_h1) if pd.notna(r.velo_h1) and pd.notna(r.velo_h2) else 0.0,
                "velo_next": np.nan, "status": "no_mlb_data",
            }
            if key_next in p.index:
                nxt = p.loc[key_next]
                if nxt.pitch_type == r.pitch_type and nxt.fb >= MIN_FB_NEXT:
                    row["velo_next"] = nxt.velo
            out.append(row)
    e = pd.DataFrame(out)
    if e.empty:
        return e
    e["is_drop"] = e["drop_mph"] <= DROP_MPH
    e["regain_mph"] = e["velo_next"] - e["velo_drop_year"]
    has = e["velo_next"].notna()
    e.loc[has, "status"] = np.where(
        e.loc[has, "regain_mph"] >= RECOVER_SHARE * (-e.loc[has, "drop_mph"]),
        "recovered", "not_recovered")
    e["recovered"] = (e["status"] == "recovered").astype(int)
    return e


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def cohens_h(p1: float, p2: float) -> float:
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def group_tests(drops: pd.DataFrame) -> pd.DataFrame:
    """Recovery rate split by each candidate explanation (Fisher exact + BH)."""
    d = drops[drops["status"] != "no_mlb_data"].copy()
    d["age_30plus"] = (d["age"] >= 30).astype(int)
    d["big_drop"] = (d["drop_mph"] <= -2.0).astype(int)
    d["faded_in_season"] = (d["fade_mph"] <= -1.0).astype(int)
    splits = {
        "absence": "Absence >=14 days / late start / early end",
        "role_to_sp": "Moved from bullpen to rotation",
        "early_down": "Velocity already down in first 5 games",
        "faded_in_season": "Lost >=1 mph from 1st to 2nd half",
        "age_30plus": "Age 30+",
        "big_drop": "Drop of 2+ mph",
    }
    rows = []
    for col, label in splits.items():
        a = d[d[col] == 1]["recovered"]
        b = d[d[col] == 0]["recovered"]
        if len(a) < 5 or len(b) < 5:
            continue
        table = [[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]]
        _, p = stats.fisher_exact(table)
        rows.append({"split": col, "label": label, "n_yes": len(a), "n_no": len(b),
                     "recovery_yes": a.mean(), "recovery_no": b.mean(),
                     "diff_pp": 100 * (a.mean() - b.mean()),
                     "cohens_h": cohens_h(a.mean(), b.mean()), "p": p})
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["q_bh"] = benjamini_hochberg(res["p"].values)
    res["label_status"] = np.where(
        (res["q_bh"] < ALPHA) & (res["diff_pp"].abs() >= MIN_EFFECT_PP),
        "CONFIRMED", "NOT CONFIRMED")
    return res


def loo_predictions(d: pd.DataFrame, features: list[str]) -> np.ndarray:
    X = d[features].astype(float).values
    y = d["recovered"].values
    pred = np.empty(len(d))
    for i in range(len(d)):
        m = np.ones(len(d), bool)
        m[i] = False
        sc = StandardScaler().fit(X[m])
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(X[m]), y[m])
        pred[i] = lr.predict_proba(sc.transform(X[i:i + 1]))[0, 1]
    return pred


def model_test(drops: pd.DataFrame, n_boot: int = 2000, seed: int = 7) -> dict:
    """LOO logistic model of recovery; AUC with paired bootstrap CI against
    a drop-size-only model."""
    d = drops[drops["status"] != "no_mlb_data"].dropna(subset=FEATURES).reset_index(drop=True)
    full = loo_predictions(d, FEATURES)
    base = loo_predictions(d, ["drop_mph"])
    y = d["recovered"].values
    rng = np.random.default_rng(seed)
    a_full, a_diff = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(d), len(d))
        if y[i].min() == y[i].max():
            continue
        af = roc_auc_score(y[i], full[i])
        a_full.append(af)
        a_diff.append(af - roc_auc_score(y[i], base[i]))
    auc = roc_auc_score(y, full)
    lo, hi = np.percentile(a_full, [2.5, 97.5])
    dlo, dhi = np.percentile(a_diff, [2.5, 97.5])
    sc = StandardScaler().fit(d[FEATURES].astype(float))
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(d[FEATURES].astype(float)), y)
    return {
        "n": len(d), "auc": auc, "auc_ci": (lo, hi),
        "auc_drop_only": roc_auc_score(y, base), "auc_gain_ci": (dlo, dhi),
        "coefs": dict(zip(FEATURES, lr.coef_[0])),
        "label": "CONFIRMED" if (lo > 0.5 and auc >= MIN_AUC) else "NOT CONFIRMED",
        "model": (sc, lr), "pred": full, "data": d,
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / s
