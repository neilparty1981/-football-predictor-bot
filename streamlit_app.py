from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from football_bot.data import download_fixtures, download_history, select_fixture_window
from football_bot.leagues import DEFAULT_LEAGUES, LEAGUES
from football_bot.learning import SelfLearningCalibrator
from football_bot.model import LeaguePoissonModel
from football_bot.predict import predict_fixtures, strongest_market

st.set_page_config(page_title="FootballBot V2.1", page_icon="⚽", layout="wide")


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


def add_strongest_market(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out[["StrongestMarket", "StrongestProb"]] = out.apply(
        lambda r: pd.Series(strongest_market(r)), axis=1
    )
    return out


def prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    display = df[[
        "Date", "League", "HomeTeam", "AwayTeam", "ResultPick", "Home", "Draw", "Away",
        "Over1.5", "Over2.5", "Over3.5", "BTTS", "P_1-1", "MostLikelyScore",
        "StrongestMarket", "StrongestProb"
    ]].copy()
    display.columns = [
        "Date", "League", "Home team", "Away team", "Result pick", "Home %", "Draw %", "Away %",
        "O1.5 %", "O2.5 %", "O3.5 %", "BTTS %", "1-1 %", "Likely score",
        "Strongest market", "Strongest %"
    ]
    display["Date"] = pd.to_datetime(display["Date"]).dt.strftime("%a %d %b")
    for c in ["Home %", "Draw %", "Away %", "O1.5 %", "O2.5 %", "O3.5 %", "BTTS %", "1-1 %", "Strongest %"]:
        display[c] = display[c].map(pct)
    return display


st.title("⚽ Football Predictor Bot V2.1")
st.caption("Today's predictions • Premier League score calls • 1X2 • Goals • BTTS • Top 5 likely 1–1s")

with st.sidebar:
    st.header("Filters")
    league_options = list(LEAGUES.keys())
    selected = st.multiselect(
        "Leagues",
        options=league_options,
        default=DEFAULT_LEAGUES,
        format_func=lambda x: LEAGUES[x],
    )
    days = st.slider("Fixtures ahead", 1, 21, 7)
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

# Always include the Premier League internally so the dedicated score tab works
# even if the user removes E0 from the general league filter.
selected_leagues = tuple(selected)
model_leagues = tuple(dict.fromkeys([*selected_leagues, "E0"]))

try:
    with st.spinner("Downloading results and retraining the model..."):
        model, learner, history_count, latest_result = train_models(
            model_leagues, seasons, float(half_life), float(shrink_games)
        )
        fixture_feed = select_fixture_window(load_fixtures(model_leagues), days=days)
except Exception as exc:
    st.error(f"Could not update FootballBot: {exc}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Training matches", f"{history_count:,}")
c2.metric("Latest result in feed", latest_result.strftime("%d %b %Y") if pd.notna(latest_result) else "—")
c3.metric("Validation matches", f"{learner.validation_matches:,}")
c4.metric("Last app refresh", datetime.now().strftime("%H:%M"))

if fixture_feed.empty:
    st.info("No fixtures are currently listed in the selected date window by the data feed.")
    st.stop()

all_pred = predict_fixtures(model, fixture_feed, calibrator=learner)
if all_pred.empty:
    st.info("No predictions could be generated for the current fixture list.")
    st.stop()

all_pred = add_strongest_market(all_pred)
all_pred = all_pred.sort_values(["Date", "League", "HomeTeam"]).reset_index(drop=True)

# Main prediction set follows the user's league filter.
pred = all_pred[all_pred["Div"].isin(selected_leagues)].copy()
epl_pred = all_pred[all_pred["Div"] == "E0"].copy()

today = pd.Timestamp.now().normalize()
today_pred = pred[pd.to_datetime(pred["Date"]).dt.normalize() == today].copy()

(
    today_tab,
    epl_tab,
    main_tab,
    oneone_tab,
    best_tab,
    performance_tab,
    about_tab,
) = st.tabs([
    "☀️ Today",
    "🏆 Premier League scores",
    "📋 Fixtures",
    "🎯 Top 5 1–1",
    "🔥 Best Markets",
    "🧠 Learning",
    "ℹ️ About",
])

with today_tab:
    st.subheader(f"Today's predictions — {today.strftime('%a %d %b %Y')}")
    if today_pred.empty:
        st.info("There are no fixtures in today's FootballBot feed for the leagues you selected.")
        future = pred[pd.to_datetime(pred["Date"]).dt.normalize() > today]
        if not future.empty:
            next_day = pd.to_datetime(future["Date"]).dt.normalize().min()
            st.caption(f"Next fixture date currently available: {next_day.strftime('%a %d %b %Y')}")
    else:
        best_home = today_pred.nlargest(1, "Home").iloc[0]
        best_o25 = today_pred.nlargest(1, "Over2.5").iloc[0]
        best_btts = today_pred.nlargest(1, "BTTS").iloc[0]
        best_11 = today_pred.nlargest(1, "P_1-1").iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Best home win", f"{best_home['HomeTeam']} {pct(best_home['Home'])}")
        m2.metric("Best Over 2.5", f"{best_o25['HomeTeam']} v {best_o25['AwayTeam']}", pct(best_o25["Over2.5"]))
        m3.metric("Best BTTS", f"{best_btts['HomeTeam']} v {best_btts['AwayTeam']}", pct(best_btts["BTTS"]))
        m4.metric("Best 1–1", f"{best_11['HomeTeam']} v {best_11['AwayTeam']}", pct(best_11["P_1-1"]))

        st.dataframe(prediction_table(today_pred), use_container_width=True, hide_index=True, height=560)

with epl_tab:
    st.subheader("English Premier League — predicted score for every listed game")
    st.caption(f"Showing Premier League fixtures in the next {days} day(s). Change 'Fixtures ahead' in the sidebar to extend the list.")
    if epl_pred.empty:
        st.info("No Premier League fixtures are currently listed in this date window.")
    else:
        score_view = epl_pred[[
            "Date", "HomeTeam", "AwayTeam", "MostLikelyScore", "xG_H", "xG_A",
            "Home", "Draw", "Away", "Over2.5", "BTTS", "P_1-1"
        ]].copy()
        score_view.columns = [
            "Date", "Home team", "Away team", "Predicted score", "Home xG", "Away xG",
            "Home %", "Draw %", "Away %", "O2.5 %", "BTTS %", "1-1 %"
        ]
        score_view["Date"] = pd.to_datetime(score_view["Date"]).dt.strftime("%a %d %b")
        score_view["Home xG"] = score_view["Home xG"].map(lambda x: f"{x:.2f}")
        score_view["Away xG"] = score_view["Away xG"].map(lambda x: f"{x:.2f}")
        for c in ["Home %", "Draw %", "Away %", "O2.5 %", "BTTS %", "1-1 %"]:
            score_view[c] = score_view[c].map(pct)
        st.dataframe(score_view, use_container_width=True, hide_index=True, height=620)

with main_tab:
    st.subheader("Upcoming predictions")
    strong_only = st.toggle(f"Only show selections ≥ {pct(min_prob)}", value=False)
    view = pred[pred["StrongestProb"] >= min_prob].copy() if strong_only else pred.copy()
    st.dataframe(prediction_table(view), use_container_width=True, hide_index=True, height=620)

    csv = pred.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions CSV", csv, "footballbot_predictions.csv", "text/csv")

with oneone_tab:
    st.subheader("Top 5 most likely 1–1 results")
    top = pred.nlargest(5, "P_1-1")
    for i, (_, r) in enumerate(top.iterrows(), 1):
        st.markdown(
            f"**{i}. {r['HomeTeam']} v {r['AwayTeam']}**  \n"
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
        * You do **not** need to upload historical CSV files for the standard model: FootballBot downloads its historical league data automatically.

        **Optional custom CSV data**

        In a future version we can add a CSV uploader for extra/private data such as xG, shots, corners, injuries, closing odds or your own archived predictions. That would supplement the automatic feed rather than replace it.

        This is statistical decision support, not a guarantee of betting profit. Backtest it and compare fair odds with live prices before staking money.
        """
    )
