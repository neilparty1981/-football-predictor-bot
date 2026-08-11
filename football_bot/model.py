from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial, log

import numpy as np
import pandas as pd


@dataclass
class TeamStrength:
    home_attack: float = 1.0
    home_defence: float = 1.0
    away_attack: float = 1.0
    away_defence: float = 1.0
    recent_attack: float = 1.0
    recent_defence: float = 1.0


class LeaguePoissonModel:
    def __init__(self, half_life_days: float = 180.0, shrink_games: float = 8.0, max_goals: int = 8):
        self.half_life_days = half_life_days
        self.shrink_games = shrink_games
        self.max_goals = max_goals
        self.models: dict[str, dict] = {}

    @staticmethod
    def _shrink(raw: float, n_eff: float, prior: float = 1.0, k: float = 8.0) -> float:
        return (raw * n_eff + prior * k) / (n_eff + k) if (n_eff + k) > 0 else prior

    def fit(self, history: pd.DataFrame) -> "LeaguePoissonModel":
        hist = history.copy()
        hist = hist.dropna(subset=["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        now = hist["Date"].max()
        if pd.isna(now):
            raise ValueError("No valid dates in historical data")

        age_days = (now - hist["Date"]).dt.days.clip(lower=0)
        hist["w"] = np.exp(-np.log(2) * age_days / self.half_life_days)

        for div, d in hist.groupby("Div"):
            if len(d) < 20:
                continue
            total_w = d["w"].sum()
            lg_home = float((d["FTHG"] * d["w"]).sum() / total_w)
            lg_away = float((d["FTAG"] * d["w"]).sum() / total_w)
            lg_home = max(lg_home, 0.25)
            lg_away = max(lg_away, 0.25)

            teams = sorted(set(d["HomeTeam"]).union(d["AwayTeam"]))
            strengths: dict[str, TeamStrength] = {}

            for team in teams:
                h = d[d["HomeTeam"] == team]
                a = d[d["AwayTeam"] == team]

                hw = h["w"].sum()
                aw = a["w"].sum()

                h_for = (h["FTHG"] * h["w"]).sum() / hw if hw else lg_home
                h_against = (h["FTAG"] * h["w"]).sum() / hw if hw else lg_away
                a_for = (a["FTAG"] * a["w"]).sum() / aw if aw else lg_away
                a_against = (a["FTHG"] * a["w"]).sum() / aw if aw else lg_home

                ha = self._shrink(h_for / lg_home, hw, k=self.shrink_games)
                hd = self._shrink(h_against / lg_away, hw, k=self.shrink_games)
                aa = self._shrink(a_for / lg_away, aw, k=self.shrink_games)
                ad = self._shrink(a_against / lg_home, aw, k=self.shrink_games)

                # Recent all-venue form: final 8 games with a moderate impact only.
                recent_h = h[["Date", "FTHG", "FTAG"]].rename(columns={"FTHG": "GF", "FTAG": "GA"})
                recent_a = a[["Date", "FTHG", "FTAG"]].rename(columns={"FTAG": "GF", "FTHG": "GA"})
                recent = pd.concat([recent_h, recent_a]).sort_values("Date").tail(8)
                if len(recent):
                    r_for = recent["GF"].mean() / max((lg_home + lg_away) / 2, 0.25)
                    r_against = recent["GA"].mean() / max((lg_home + lg_away) / 2, 0.25)
                    ra = self._shrink(r_for, len(recent), k=6.0)
                    rd = self._shrink(r_against, len(recent), k=6.0)
                else:
                    ra = rd = 1.0

                strengths[team] = TeamStrength(ha, hd, aa, ad, ra, rd)

            self.models[str(div)] = {
                "home_avg": lg_home,
                "away_avg": lg_away,
                "strengths": strengths,
            }
        return self

    def expected_goals(self, div: str, home: str, away: str) -> tuple[float, float]:
        m = self.models.get(str(div))
        if not m:
            raise KeyError(f"No trained model for league {div}")
        hs = m["strengths"].get(home, TeamStrength())
        aws = m["strengths"].get(away, TeamStrength())

        # Core venue-specific attack/defence with a modest recent-form multiplier.
        xh = m["home_avg"] * hs.home_attack * aws.away_defence
        xa = m["away_avg"] * aws.away_attack * hs.home_defence
        xh *= (hs.recent_attack * aws.recent_defence) ** 0.18
        xa *= (aws.recent_attack * hs.recent_defence) ** 0.18

        return float(np.clip(xh, 0.15, 4.5)), float(np.clip(xa, 0.15, 4.5))

    @staticmethod
    def _pois(k: int, lam: float) -> float:
        return exp(-lam) * (lam ** k) / factorial(k)

    def probabilities(self, div: str, home: str, away: str) -> dict[str, float | str]:
        xh, xa = self.expected_goals(div, home, away)
        n = self.max_goals
        hp = np.array([self._pois(i, xh) for i in range(n + 1)])
        ap = np.array([self._pois(j, xa) for j in range(n + 1)])
        matrix = np.outer(hp, ap)
        matrix /= matrix.sum()

        p_home = float(np.tril(matrix, -1).sum())
        p_draw = float(np.trace(matrix))
        p_away = float(np.triu(matrix, 1).sum())

        p_o15 = float(sum(matrix[i, j] for i in range(n + 1) for j in range(n + 1) if i + j >= 2))
        p_o25 = float(sum(matrix[i, j] for i in range(n + 1) for j in range(n + 1) if i + j >= 3))
        p_o35 = float(sum(matrix[i, j] for i in range(n + 1) for j in range(n + 1) if i + j >= 4))
        p_btts = float(matrix[1:, 1:].sum())
        p_11 = float(matrix[1, 1])

        best_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        best_score = f"{best_idx[0]}-{best_idx[1]}"
        result_probs = {"Home": p_home, "Draw": p_draw, "Away": p_away}
        pick = max(result_probs, key=result_probs.get)
        best_prob = result_probs[pick]

        # Normalised confidence derived from 1X2 entropy (0=uncertain, 1=very concentrated).
        probs = np.array([p_home, p_draw, p_away])
        entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1)))) / log(3)
        confidence = 1.0 - entropy

        return {
            "xG_H": xh,
            "xG_A": xa,
            "Home": p_home,
            "Draw": p_draw,
            "Away": p_away,
            "ResultPick": pick,
            "ResultProb": best_prob,
            "Over1.5": p_o15,
            "Over2.5": p_o25,
            "Over3.5": p_o35,
            "BTTS": p_btts,
            "P_1-1": p_11,
            "MostLikelyScore": best_score,
            "Confidence": confidence,
        }
