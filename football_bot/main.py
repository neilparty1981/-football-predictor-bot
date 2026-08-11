from __future__ import annotations

import argparse
import sys

import pandas as pd

from .data import download_fixtures, download_history, select_fixture_window
from .leagues import DEFAULT_LEAGUES, LEAGUES
from .learning import SelfLearningCalibrator
from .model import LeaguePoissonModel
from .predict import predict_fixtures, strongest_market


def pct(x: float) -> str:
    return f"{100*x:.1f}%"


def cmd_predict(args: argparse.Namespace) -> int:
    leagues = args.league or DEFAULT_LEAGUES
    unknown = [x for x in leagues if x not in LEAGUES]
    if unknown:
        print(f"Unknown league code(s): {', '.join(unknown)}")
        print("Known codes:", ", ".join(LEAGUES))
        return 2

    print("Downloading historical results...")
    history = download_history(leagues=leagues, seasons=args.seasons)
    print(f"Loaded {len(history):,} completed matches.")

    print("Training self-learning calibration...")
    learner = SelfLearningCalibrator().fit_from_history(
        history,
        half_life_days=args.half_life,
        shrink_games=args.shrink_games,
    )

    print("Training final league models...")
    model = LeaguePoissonModel(
        half_life_days=args.half_life,
        shrink_games=args.shrink_games,
    ).fit(history)

    print("Downloading upcoming fixtures...")
    fixtures = select_fixture_window(download_fixtures(leagues), days=args.days)
    if fixtures.empty:
        print("No fixtures found in that date window in Football-Data's fixture file.")
        return 0

    pred = predict_fixtures(model, fixtures, calibrator=learner)
    if pred.empty:
        print("No predictions could be produced.")
        return 0

    pred[["StrongestMarket", "StrongestProb"]] = pred.apply(
        lambda r: pd.Series(strongest_market(r)), axis=1
    )

    if args.min_prob is not None:
        pred = pred[pred["StrongestProb"] >= args.min_prob]

    show = pred.copy()
    for c in ["Home", "Draw", "Away", "Over1.5", "Over2.5", "Over3.5", "BTTS", "P_1-1", "ResultProb", "StrongestProb"]:
        show[c] = show[c].map(pct)
    show["xG"] = show["xG_H"].map(lambda x: f"{x:.2f}") + "-" + show["xG_A"].map(lambda x: f"{x:.2f}")

    cols = [
        "Date", "Div", "HomeTeam", "AwayTeam", "xG", "ResultPick", "ResultProb",
        "Home", "Draw", "Away", "Over1.5", "Over2.5", "Over3.5", "BTTS", "P_1-1", "MostLikelyScore",
        "StrongestMarket", "StrongestProb"
    ]
    print("\n=== FOOTBALL PREDICTOR V2 ===")
    print(show[cols].sort_values(["Date", "Div"]).to_string(index=False))

    top11 = pred.sort_values("P_1-1", ascending=False).head(5).copy()
    print("\n=== TOP 5 MOST LIKELY 1-1 RESULTS ===")
    for i, (_, r) in enumerate(top11.iterrows(), 1):
        print(f"{i}. {r['HomeTeam']} v {r['AwayTeam']} ({r['Div']}) — {pct(r['P_1-1'])} | fair odds {r['Fair_P_1-1']:.2f}")

    print("\n=== BEST BY MARKET ===")
    for market, label in [
        ("Home", "Home win"), ("Away", "Away win"), ("Over1.5", "Over 1.5"),
        ("Over2.5", "Over 2.5"), ("Over3.5", "Over 3.5"), ("BTTS", "BTTS Yes")
    ]:
        r = pred.sort_values(market, ascending=False).iloc[0]
        print(f"{label:12s}: {r['HomeTeam']} v {r['AwayTeam']} — {pct(r[market])}")

    if args.output:
        pred.to_csv(args.output, index=False)
        print(f"\nSaved: {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="football-bot", description="Football Predictor Bot V2")
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("predict", help="Predict upcoming fixtures")
    q.add_argument("--days", type=int, default=7, help="Upcoming days to include")
    q.add_argument("--league", nargs="+", help="League codes, e.g. E0 E1 SC0 D1 I1 SP1 F1")
    q.add_argument("--seasons", type=int, default=4, help="Historical seasons to download")
    q.add_argument("--half-life", type=float, default=180.0, help="Recency weighting half-life in days")
    q.add_argument("--shrink-games", type=float, default=8.0, help="Small-sample shrinkage strength")
    q.add_argument("--min-prob", type=float, default=None, help="Only show rows whose strongest market is at least this probability")
    q.add_argument("--output", help="Optional CSV output path")
    q.set_defaults(func=cmd_predict)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        code = args.func(args)
    except KeyboardInterrupt:
        code = 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
