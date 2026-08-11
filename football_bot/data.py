from __future__ import annotations

from datetime import datetime
from io import StringIO
from typing import Iterable

import pandas as pd
import requests

from .leagues import DEFAULT_LEAGUES

BASE = "https://www.football-data.co.uk"
UA = {"User-Agent": "Mozilla/5.0 FootballPredictorBot/1.0"}


def season_code_for_date(dt: datetime | None = None) -> str:
    dt = dt or datetime.now()
    start = dt.year if dt.month >= 7 else dt.year - 1
    end = (start + 1) % 100
    return f"{start % 100:02d}{end:02d}"


def previous_season_codes(n: int = 4, dt: datetime | None = None) -> list[str]:
    dt = dt or datetime.now()
    start = dt.year if dt.month >= 7 else dt.year - 1
    return [f"{(start-i) % 100:02d}{(start-i+1) % 100:02d}" for i in range(n)]


def _get_csv(url: str, timeout: int = 25) -> pd.DataFrame:
    r = requests.get(url, timeout=timeout, headers=UA)
    r.raise_for_status()
    text = r.content.decode("utf-8-sig", errors="replace")
    return pd.read_csv(StringIO(text), on_bad_lines="skip")


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns:
        return df
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], dayfirst=True, errors="coerce")
    return out


def download_history(
    leagues: Iterable[str] | None = None,
    seasons: int = 4,
) -> pd.DataFrame:
    leagues = list(leagues or DEFAULT_LEAGUES)
    frames: list[pd.DataFrame] = []

    for season in previous_season_codes(seasons):
        for league in leagues:
            url = f"{BASE}/mmz4281/{season}/{league}.csv"
            try:
                df = _get_csv(url)
            except Exception:
                continue
            if not {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}.issubset(df.columns):
                continue
            df = _parse_dates(df)
            df["Div"] = df.get("Div", league).fillna(league)
            df["SeasonCode"] = season
            df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
            frames.append(df)

    if not frames:
        raise RuntimeError("No historical data could be downloaded. Check internet access or the league codes.")

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["FTHG"] = pd.to_numeric(out["FTHG"], errors="coerce")
    out["FTAG"] = pd.to_numeric(out["FTAG"], errors="coerce")
    return out.dropna(subset=["FTHG", "FTAG"])


def download_fixtures(leagues: Iterable[str] | None = None) -> pd.DataFrame:
    leagues = set(leagues or DEFAULT_LEAGUES)
    df = _get_csv(f"{BASE}/fixtures.csv")
    df = _parse_dates(df)
    required = {"Div", "Date", "HomeTeam", "AwayTeam"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Fixture file is missing required columns: {sorted(required - set(df.columns))}")
    return df[df["Div"].isin(leagues)].copy()


def select_fixture_window(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    today = pd.Timestamp.now().normalize()
    end = today + pd.Timedelta(days=days)
    return df[(df["Date"] >= today) & (df["Date"] <= end)].copy()
