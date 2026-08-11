from __future__ import annotations

from dataclasses import dataclass, asdict
from math import log
from typing import Iterable

import numpy as np
import pandas as pd

from .model import LeaguePoissonModel

BINARY_MARKETS = ["Home", "Draw", "Away", "Over1.5", "Over2.5", "Over3.5", "BTTS", "P_1-1"]


def _clip_prob(p: np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)


def _logit(p: np.ndarray | float) -> np.ndarray:
    p = _clip_prob(p)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray | float) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=float), -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class LogisticCalibrator:
    intercept: float = 0.0
    slope: float = 1.0
    fitted: bool = False

    def fit(
        self,
        probabilities: Iterable[float],
        outcomes: Iterable[int],
        iterations: int = 700,
        learning_rate: float = 0.035,
        l2: float = 0.015,
    ) -> "LogisticCalibrator":
        p = _clip_prob(np.asarray(list(probabilities), dtype=float))
        y = np.asarray(list(outcomes), dtype=float)
        if len(p) < 60 or len(np.unique(y)) < 2:
            return self

        x = _logit(p)
        a, b = 0.0, 1.0
        n = float(len(y))

        for _ in range(iterations):
            pred = _sigmoid(a + b * x)
            err = pred - y
            grad_a = float(err.mean()) + l2 * a / n
            grad_b = float((err * x).mean()) + l2 * (b - 1.0) / n
            a -= learning_rate * grad_a
            b -= learning_rate * grad_b

        self.intercept = float(np.clip(a, -2.5, 2.5))
        self.slope = float(np.clip(b, 0.25, 2.5))
        self.fitted = True
        return self

    def transform(self, p: float) -> float:
        if not self.fitted:
            return float(p)
        z = self.intercept + self.slope * float(_logit(p))
        return float(_sigmoid(z))


class SelfLearningCalibrator:
    """Probability calibration trained on a chronological holdout sample.

    This is deliberately conservative: it learns how over/under-confident the
    base Poisson model has been, rather than blindly chasing the last result.
    """

    def __init__(self) -> None:
        self.calibrators = {market: LogisticCalibrator() for market in BINARY_MARKETS}
        self.metrics: dict[str, dict[str, float]] = {}
        self.validation_matches = 0
        self.cutoff: pd.Timestamp | None = None

    @staticmethod
    def outcomes(row: pd.Series) -> dict[str, int]:
        hg = int(row["FTHG"])
        ag = int(row["FTAG"])
        total = hg + ag
        return {
            "Home": int(hg > ag),
            "Draw": int(hg == ag),
            "Away": int(hg < ag),
            "Over1.5": int(total >= 2),
            "Over2.5": int(total >= 3),
            "Over3.5": int(total >= 4),
            "BTTS": int(hg > 0 and ag > 0),
            "P_1-1": int(hg == 1 and ag == 1),
        }

    def fit_from_history(
        self,
        history: pd.DataFrame,
        half_life_days: float = 180.0,
        shrink_games: float = 8.0,
        validation_fraction: float = 0.20,
    ) -> "SelfLearningCalibrator":
        hist = history.sort_values("Date").reset_index(drop=True).copy()
        if len(hist) < 300:
            return self

        # Keep the split chronological to avoid training on future matches.
        split = max(200, int(len(hist) * (1 - validation_fraction)))
        split = min(split, len(hist) - 80)
        train = hist.iloc[:split].copy()
        valid = hist.iloc[split:].copy()
        self.validation_matches = len(valid)
        self.cutoff = valid["Date"].min() if len(valid) else None

        base_model = LeaguePoissonModel(
            half_life_days=half_life_days,
            shrink_games=shrink_games,
        ).fit(train)

        rows: list[dict] = []
        for _, r in valid.iterrows():
            try:
                probs = base_model.probabilities(str(r["Div"]), str(r["HomeTeam"]), str(r["AwayTeam"]))
            except (KeyError, ValueError):
                continue
            actual = self.outcomes(r)
            rows.append({**{m: float(probs[m]) for m in BINARY_MARKETS}, **{f"Y_{m}": actual[m] for m in BINARY_MARKETS}})

        if not rows:
            return self

        df = pd.DataFrame(rows)
        self.validation_matches = len(df)

        for market in BINARY_MARKETS:
            p = df[market].astype(float).to_numpy()
            y = df[f"Y_{market}"].astype(int).to_numpy()
            cal = LogisticCalibrator().fit(p, y)
            self.calibrators[market] = cal
            p_cal = np.array([cal.transform(x) for x in p])
            before = float(np.mean((p - y) ** 2))
            after = float(np.mean((p_cal - y) ** 2))
            self.metrics[market] = {
                "brier_before": before,
                "brier_after": after,
                "improvement": before - after,
                "samples": float(len(y)),
                "base_rate": float(y.mean()),
            }
        return self

    def calibrate_prediction(self, pred: dict[str, float | str]) -> dict[str, float | str]:
        out = dict(pred)

        # Calibrate 1X2 one-vs-rest then renormalise to a valid probability vector.
        result_vals = []
        for market in ["Home", "Draw", "Away"]:
            result_vals.append(self.calibrators[market].transform(float(pred[market])))
        total = sum(result_vals) or 1.0
        for market, value in zip(["Home", "Draw", "Away"], result_vals):
            out[market] = value / total

        for market in ["Over1.5", "Over2.5", "Over3.5", "BTTS", "P_1-1"]:
            out[market] = self.calibrators[market].transform(float(pred[market]))

        result_probs = {m: float(out[m]) for m in ["Home", "Draw", "Away"]}
        pick = max(result_probs, key=result_probs.get)
        out["ResultPick"] = pick
        out["ResultProb"] = result_probs[pick]

        # Confidence after calibration, based on concentration of 1X2 probabilities.
        probs = np.array(list(result_probs.values()), dtype=float)
        entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1)))) / log(3)
        out["Confidence"] = 1.0 - entropy
        return out

    def metrics_frame(self) -> pd.DataFrame:
        rows = []
        for market, m in self.metrics.items():
            rows.append({
                "Market": market,
                "Brier before": m["brier_before"],
                "Brier after": m["brier_after"],
                "Improvement": m["improvement"],
                "Validation matches": int(m["samples"]),
                "Observed rate": m["base_rate"],
                "Calibrator fitted": self.calibrators[market].fitted,
            })
        return pd.DataFrame(rows)

    def as_dict(self) -> dict:
        return {
            "validation_matches": self.validation_matches,
            "cutoff": self.cutoff.isoformat() if self.cutoff is not None else None,
            "calibrators": {k: asdict(v) for k, v in self.calibrators.items()},
            "metrics": self.metrics,
        }
