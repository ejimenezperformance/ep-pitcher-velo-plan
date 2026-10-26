"""Known-answer tests, in both directions, before touching real data.

Run: python tests/test_synthetic.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import velo  # noqa: E402


def fake_games(pid, year, velo_mph, n_games=30, start="04-05", every=6, total=20, ptype="FF"):
    dates = pd.date_range(f"{year}-{start}", periods=n_games, freq=f"{every}D")
    return pd.DataFrame({"player_id": pid, "player_name": f"P{pid}", "game_date": dates,
                         "game_pk": range(n_games), "pitches": 12, "total_pitches": total,
                         "velocity": velo_mph, "pitch_type": ptype})


def test_panel_and_events():
    # Pitcher 1: 95 -> 93 -> 94.5 (drop 2, regains 1.5 = recovered)
    # Pitcher 2: 95 -> 93 -> 93.2 (drop 2, regains 0.2 = not recovered)
    # Pitcher 3: 95 -> 94.8      (no drop)
    # Pitcher 4: drop with a 30-day gap -> absence flag
    g = pd.concat([
        fake_games(1, 2021, 95), fake_games(1, 2022, 93), fake_games(1, 2023, 94.5),
        fake_games(2, 2021, 95), fake_games(2, 2022, 93), fake_games(2, 2023, 93.2),
        fake_games(3, 2021, 95), fake_games(3, 2022, 94.8), fake_games(3, 2023, 94.8),
        fake_games(4, 2021, 95), fake_games(4, 2022, 93, every=6).assign(
            game_date=lambda d: d["game_date"].where(d.index < 10, d["game_date"] + pd.Timedelta(days=30))),
    ])
    p = velo.season_panel(g)
    e = velo.drop_events(p, [2022]).set_index("player_id")
    assert abs(e.loc[1, "drop_mph"] + 2) < 1e-9
    assert e.loc[1, "status"] == "recovered"
    assert e.loc[2, "status"] == "not_recovered"
    assert not e.loc[3, "is_drop"]
    assert e.loc[4, "absence"] == 1 and e.loc[1, "absence"] == 0
    assert e.loc[4, "status"] == "no_mlb_data"
    # Mixed pitch type must not count: switch FF -> SI
    g2 = pd.concat([fake_games(5, 2021, 95), fake_games(5, 2022, 92, ptype="SI")])
    assert velo.drop_events(velo.season_panel(g2), [2022]).empty
    print("PASS panel/events")


def fake_drops(n, effect, seed):
    """effect > 0: absence raises recovery; < 0: lowers it; 0: no link."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "drop_mph": rng.uniform(-3, -1, n), "age": rng.integers(23, 36, n),
        "absence": rng.integers(0, 2, n), "role_to_sp": rng.integers(0, 2, n),
        "early_down": rng.integers(0, 2, n), "fade_mph": rng.normal(0, 0.5, n),
        "fb_base": rng.integers(300, 1500, n), "status": "x",
    })
    prob = np.clip(0.45 + effect * (d["absence"] - 0.5), 0.02, 0.98)
    d["recovered"] = (rng.uniform(size=n) < prob).astype(int)
    return d


def test_group_and_model():
    for effect, expect in [(0.6, "CONFIRMED"), (-0.6, "CONFIRMED"), (0.0, "NOT CONFIRMED")]:
        d = fake_drops(400, effect, seed=1)
        g = velo.group_tests(d).set_index("split")
        got = g.loc["absence", "label_status"]
        assert got == expect, (effect, got)
        if effect != 0:
            assert np.sign(g.loc["absence", "diff_pp"]) == np.sign(effect)
        others = g.drop(index="absence")["label_status"]
        assert (others == "NOT CONFIRMED").all(), others
        m = velo.model_test(d, n_boot=300)
        assert m["label"] == expect, (effect, m["auc"], m["auc_ci"])
        print(f"PASS effect={effect:+.1f}: absence {got}, model AUC {m['auc']:.2f} {m['label']}")


if __name__ == "__main__":
    test_panel_and_events()
    test_group_and_model()
    print("ALL SYNTHETIC TESTS PASSED")
