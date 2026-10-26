"""Step 1: what happens to a fastball velocity drop with no known intervention?

Builds the season panel (2017-2026), finds every >=1 mph drop in event years
2018, 2022-2025, and measures recovery the following season.
Outputs go to outputs/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import velo

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "outputs"


def load_panel() -> pd.DataFrame:
    cache = OUT / "season_panel.csv"
    if cache.exists():
        return pd.read_csv(cache)
    games = pd.concat([pd.read_csv(f) for f in sorted(RAW.glob("fastballs_*.csv"))])
    ages = pd.read_csv(RAW / "ages.csv")
    p = velo.season_panel(games, ages)
    p.to_csv(cache, index=False, encoding="utf-8")
    return p


def main():
    OUT.mkdir(exist_ok=True)
    panel = load_panel()
    ev = velo.drop_events(panel, velo.EVENT_YEARS)
    ev.to_csv(OUT / "all_pairs.csv", index=False, encoding="utf-8")
    drops = ev[ev["is_drop"]].copy()
    drops.to_csv(OUT / "baseline_drops.csv", index=False, encoding="utf-8")

    # Comparison line: stable pitchers (|change| < 0.5 mph) — what happens next year
    stable = ev[(ev["drop_mph"].abs() < 0.5) & ev["velo_next"].notna()]
    has = drops[drops["status"] != "no_mlb_data"]

    summary = {
        "pairs_tested": int(len(ev)),
        "drops": int(len(drops)),
        "drop_share_of_pairs": float(len(drops) / len(ev)),
        "status_counts": drops["status"].value_counts().to_dict(),
        "recovery_rate_with_data": float(has["recovered"].mean()),
        "median_drop_mph": float(drops["drop_mph"].median()),
        "median_regain_mph": float(has["regain_mph"].median()),
        "stable_next_change_mean": float((stable["velo_next"] - stable["velo_drop_year"]).mean()),
        "drop_next_change_mean": float(has["regain_mph"].mean()),
        "by_year": drops.groupby("year")["status"].value_counts().unstack(fill_value=0).to_dict("index"),
    }
    # Headline test: if drops were mostly noise, about half would regain half the loss.
    k, n = int(has["recovered"].sum()), len(has)
    bt = stats.binomtest(k, n, 0.5, alternative="less")
    ci = stats.binomtest(k, n).proportion_ci(method="wilson")  # two-sided 95% CI
    summary["headline_test"] = {
        "recovered": k, "n": n, "rate": k / n, "wilson_ci": [ci.low, ci.high],
        "binomial_p_vs_50pct": bt.pvalue,
        "label": "CONFIRMED" if (bt.pvalue < velo.ALPHA and 0.5 - ci.high >= velo.MIN_EFFECT_PP / 100) else "NOT CONFIRMED",
    }
    # Does a drop come with losing a comparable MLB role next year? (drops vs stable pitchers)
    stable_all = ev[ev["drop_mph"].abs() < 0.5]
    a_out, b_out = (drops["status"] == "no_mlb_data"), (stable_all["status"] == "no_mlb_data")
    _, p_out = stats.fisher_exact([[a_out.sum(), (~a_out).sum()], [b_out.sum(), (~b_out).sum()]])
    diff = 100 * (a_out.mean() - b_out.mean())
    summary["no_next_data_vs_stable"] = {
        "drops_rate": a_out.mean(), "stable_rate": b_out.mean(), "n_stable": len(stable_all),
        "diff_pp": diff, "fisher_p": p_out, "cohens_h": velo.cohens_h(a_out.mean(), b_out.mean()),
        "label": "CONFIRMED" if (p_out < velo.ALPHA and abs(diff) >= velo.MIN_EFFECT_PP) else "NOT CONFIRMED",
    }
    # Is a drop year followed by a bigger rebound than stable pitchers get? (regression to the mean check)
    a = has["regain_mph"].values
    b = (stable["velo_next"] - stable["velo_drop_year"]).values
    summary["regain_vs_stable"] = {
        "mannwhitney_p": float(stats.mannwhitneyu(a, b).pvalue),
        "cohens_d": float(velo.cohens_d(a, b)),
    }

    groups = velo.group_tests(drops)
    groups.to_csv(OUT / "recovery_by_explanation.csv", index=False, encoding="utf-8")

    m = velo.model_test(drops)
    summary["model"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in m.items()
                        if k in ("n", "auc", "auc_ci", "auc_drop_only", "auc_gain_ci", "coefs", "label")}
    pd.DataFrame({"player_name": m["data"]["player_name"], "year": m["data"]["year"],
                  "recovered": m["data"]["recovered"], "p_recover_loo": m["pred"]}).to_csv(
        OUT / "baseline_loo_predictions.csv", index=False, encoding="utf-8")

    # Calibration in thirds of predicted probability
    d = m["data"].assign(p=m["pred"])
    d["tercile"] = pd.qcut(d["p"], 3, labels=["low", "mid", "high"])
    summary["calibration"] = d.groupby("tercile", observed=True).agg(
        n=("recovered", "size"), predicted=("p", "mean"), actual=("recovered", "mean")).round(3).to_dict("index")

    (OUT / "baseline_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))
    print(groups.round(3).to_string())


if __name__ == "__main__":
    main()
