from __future__ import annotations

import numpy as np
import pandas as pd

from .leagues import LEAGUES
from .learning import SelfLearningCalibrator
from .model import LeaguePoissonModel


def fair_odds(p: float) -> float:
    return round(1.0 / p, 2) if p > 0 else np.nan


def predict_fixtures(
    model: LeaguePoissonModel,
    fixtures: pd.DataFrame,
    calibrator: SelfLearningCalibrator | None = None,
) -> pd.DataFrame:
    rows = []
    for _, r in fixtures.iterrows():
        div = str(r["Div"])
        try:
            p = model.probabilities(div, str(r["HomeTeam"]), str(r["AwayTeam"]))
        except KeyError:
            continue
        if calibrator is not None:
            p = calibrator.calibrate_prediction(p)
        row = {
            "Date": r["Date"],
            "League": LEAGUES.get(div, div),
            "Div": div,
            "HomeTeam": r["HomeTeam"],
            "AwayTeam": r["AwayTeam"],
            **p,
        }
        for market in ["Home", "Draw", "Away", "Over1.5", "Over2.5", "Over3.5", "BTTS", "P_1-1"]:
            row[f"Fair_{market}"] = fair_odds(float(row[market]))
        rows.append(row)

    return pd.DataFrame(rows)


def strongest_market(row: pd.Series) -> tuple[str, float]:
    markets = {
        "Home Win": row["Home"],
        "Draw": row["Draw"],
        "Away Win": row["Away"],
        "Over 1.5": row["Over1.5"],
        "Over 2.5": row["Over2.5"],
        "Over 3.5": row["Over3.5"],
        "BTTS Yes": row["BTTS"],
    }
    return max(markets.items(), key=lambda kv: kv[1])
