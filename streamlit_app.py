from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from football_bot.data import download_fixtures, download_history, select_fixture_window
from football_bot.leagues import DEFAULT_LEAGUES, LEAGUES
from football_bot.learning import SelfLearningCalibrator
from football_bot.model import LeaguePoissonModel
from football_bot.predict import predict_fixtures, strongest_market

st.set_page_config(page_title="FootballBot V2", page_icon="⚽", layout="wide")


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_history(leagues: tuple[str, ...], seasons: int) -> pd.DataFrame:
    return download_history(leagues=leagues, seasons=seasons)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_fixtures(leagues: tuple[str, ...]) -> pd.DataFrame:
    return download_fixtures(leagues=leagues)


@st.cache_resource(ttl=6 * 60 * 60, show_spinner=False)
def train_models(
    leagues: tuple[str, ...],
    seasons: int,
    half_life: float,
    shrink_games: float,
):
    history = load_history(leagues, seasons)
    learner = SelfLearningCalibrator().fit_from_history(
        history,
        half_life_days=half_life,
        shrink_games=shrink_games,
    )
    final_model = LeaguePoissonModel(
        half_life_days=half_life,
        shrink_games=shrink_games,
    ).fit(history)
    return final_model, learner, len(history), history["Date"].max()


st.title("⚽ Football Predictor Bot V2")
st.caption("Self-updating probabilities • 1X2 • Goals • BTTS • Top 5 likely 1–1s")

with st.sidebar:
    st.header("Filters")
    league_options = list(LEAGUES.keys())
    selected = st.multiselect(
        "Leagues",
        options=league_options,
        default=DEFAULT_LEAGUES,
        format_func=lambda x: LEAGUES[x],
    )
    days = st.slider("Fixtures ahead", 1, 14, 7)
    seasons = st.slider("Training seasons", 2, 6, 4)
    min_prob = st.slider("Highlight threshold", 50, 90, 65) / 100

    with st.expander("Model settings"):
        half_life = st.slider("Recency half-life (days)", 60, 365, 180, 15)
        shrink_games = st.slider("Small-sample shrinkage", 3, 20, 8)

    if st.button("🔄 Refresh latest data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

if not selected:
    st.warning("Choose at least one league.")
    st.stop()

leagues = tuple(selected)

try:
    with st.spinner("Downloading results and retraining the model..."):
        model, learner, history_count, latest_result = train_models(
            leagues, seasons, float(half_life), float(shrink_games)
        )
        fixtures = select_fixture_window(load_fixtures(leagues), days=days)
except Exception as exc:
    st.error(f"Could not update FootballBot: {exc}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Training matches", f"{history_count:,}")
c2.metric("Latest result in feed", latest_result.strftime("%d %b %Y") if pd.notna(latest_result) else "—")
c3.metric("Validation matches", f"{learner.validation_matches:,}")
c4.metric("Last app refresh", datetime.now().strftime("%H:%M"))

if fixtures.empty:
    st.info("No fixtures are currently listed in the selected date window by the data feed.")
    st.stop()

pred = predict_fixtures(model, fixtures, calibrator=learner)
if pred.empty:
    st.info("No predictions could be generated for the current fixture list.")
    st.stop()

pred[["StrongestMarket", "StrongestProb"]] = pred.apply(
    lambda r: pd.Series(strongest_market(r)), axis=1
)

# Friendly date formatting without destroying sortable raw data.
pred = pred.sort_values(["Date", "League", "HomeTeam"]).reset_index(drop=True)

main_tab, oneone_tab, best_tab, performance_tab, about_tab = st.tabs(
    ["📋 Fixtures", "🎯 Top 5 1–1", "🔥 Best Markets", "🧠 Learning", "ℹ️ About"]
)

with main_tab:
    st.subheader("Upcoming predictions")
    strong_only = st.toggle(f"Only show selections ≥ {pct(min_prob)}", value=False)
    view = pred[pred["StrongestProb"] >= min_prob].copy() if strong_only else pred.copy()

    display = view[[
        "Date", "League", "HomeTeam", "AwayTeam", "ResultPick", "Home", "Draw", "Away",
        "Over1.5", "Over2.5", "Over3.5", "BTTS", "P_1-1", "MostLikelyScore", "StrongestMarket", "StrongestProb"
    ]].copy()
    display.columns = [
        "Date", "League", "Home team", "Away team", "Result pick", "Home %", "Draw %", "Away %",
        "O1.5 %", "O2.5 %", "O3.5 %", "BTTS %", "1-1 %", "Likely score", "Strongest market", "Strongest %"
    ]
    display["Date"] = pd.to_datetime(display["Date"]).dt.strftime("%a %d %b")
    for c in ["Home %", "Draw %", "Away %", "O1.5 %", "O2.5 %", "O3.5 %", "BTTS %", "1-1 %", "Strongest %"]:
        display[c] = display[c].map(pct)
    st.dataframe(display, use_container_width=True, hide_index=True, height=620)

    csv = pred.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions CSV", csv, "footballbot_predictions.csv", "text/csv")

with oneone_tab:
    st.subheader("Top 5 most likely 1–1 results")
    top = pred.nlargest(5, "P_1-1")
    for i, (_, r) in enumerate(top.iterrows(), 1):
        st.markdown(
            f"**{i}. {r['Home']} v {r['Away']}**  \\n"
            f"{r['League']} • {pd.to_datetime(r['Date']).strftime('%a %d %b')} • "
            f"**1–1 {pct(r['P_1-1'])}** • model fair odds **{r['Fair_P_1-1']:.2f}**"
        )

with best_tab:
    st.subheader("Highest model probability by market")
    markets = [
        ("Home", "Home win"), ("Draw", "Draw"), ("Away", "Away win"),
        ("Over1.5", "Over 1.5"), ("Over2.5", "Over 2.5"), ("Over3.5", "Over 3.5"),
        ("BTTS", "BTTS Yes"),
    ]
    rows = []
    for market, label in markets:
        r = pred.nlargest(1, market).iloc[0]
        rows.append({
            "Market": label,
            "Match": f"{r['HomeTeam']} v {r['AwayTeam']}",
            "League": r["League"],
            "Probability": pct(r[market]),
            "Fair odds": r[f"Fair_{market}"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

with performance_tab:
    st.subheader("Self-learning calibration")
    st.write(
        "FootballBot trains a base scoring model on older matches, tests it on a chronological holdout set, "
        "then learns conservative probability calibration from those outcomes. The final fixture model is retrained "
        "on all available completed matches each refresh."
    )
    metrics = learner.metrics_frame()
    if metrics.empty:
        st.info("Not enough validation data to fit calibration yet.")
    else:
        show_metrics = metrics.copy()
        for c in ["Brier before", "Brier after", "Improvement", "Observed rate"]:
            show_metrics[c] = show_metrics[c].map(lambda x: f"{x:.4f}")
        st.dataframe(show_metrics, hide_index=True, use_container_width=True)
        improved = int((metrics["Improvement"] > 0).sum())
        st.caption(f"Calibration improved holdout Brier score in {improved}/{len(metrics)} tracked markets. Lower Brier is better.")

with about_tab:
    st.markdown(
        """
        **What 'self-learning' means here**

        * New completed results are downloaded automatically when the cache refreshes.
        * Team attack/defence and recent-form strengths are retrained from the latest data.
        * A calibration layer learns whether the raw model has historically been too confident or too cautious.
        * Validation is chronological to reduce future-data leakage.

        **Phone use**

        Deploy this repository to Streamlit Community Cloud. You then open the `streamlit.app` link in Safari/Chrome and can add it to your phone Home Screen.

        This is statistical decision support, not a guarantee of betting profit. Backtest it and compare fair odds with live prices before staking money.
        """
    )
