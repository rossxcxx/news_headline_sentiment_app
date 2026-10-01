"""
Headlyn
A GenAI-powered Streamlit app for reading the tone of the news.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_data, get_filtered_data
from src.genai_analysis import (
    classify_sentiment_batch,
    answer_question_about_data,
    GenAIUnavailableError,
)

st.set_page_config(
    page_title="Headlyn",
    page_icon="📰",
    layout="wide",
)

def _md_safe(text) -> str:
    """Escape characters that Streamlit's markdown would otherwise interpret (e.g. $ as LaTeX)."""
    return str(text).replace("$", "\\$")


# Analysis always runs in GenAI (Hugging Face API) mode — the offline/demo
# toggle has been removed from the UI, so this is fixed rather than user-chosen.
analysis_mode = "genai"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📰 Headlyn")
st.caption(
    "Headline sentiment & trend explorer — run GenAI-powered sentiment "
    "(tone) analysis and keyword extraction on news headlines, and explore trends interactively."
)

# ---------------------------------------------------------------------------
# Home screen: Dataset panel (left) + Filters & Run controls (right)
# ---------------------------------------------------------------------------
default_path = "data/news_headlines_sample.csv"

with st.container(border=True):
    dataset_col, filters_col = st.columns([1, 3])

    with dataset_col:
        st.subheader("Dataset")

        try:
            df = load_data(default_path)
        except Exception as e:
            st.error(f"Could not load dataset: {e}")
            st.stop()

        st.success(f"Loaded {len(df):,} headlines from the bundled dataset.")

    with filters_col:
        st.subheader("Filters")
        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            category_options = sorted(df["category"].dropna().unique().tolist())
            selected_categories = st.multiselect("Category(ies)", category_options, default=[])

        with f_col2:
            author_options = sorted(df["authors"].dropna().unique().tolist())
            selected_authors = st.multiselect("Author(s)", author_options, default=[])

        date_range = None
        with f_col3:
            if df["date"].notna().any():
                min_date = df["date"].min().date()
                max_date = df["date"].max().date()
                date_range = st.date_input("Publication date range", (min_date, max_date))
                if isinstance(date_range, tuple) and len(date_range) != 2:
                    date_range = None

        filtered_df = get_filtered_data(
            df,
            categories=selected_categories or None,
            authors=selected_authors or None,
            date_range=date_range,
        )

        st.markdown(f"**{len(filtered_df):,}** headlines match current filters.")

        # Run Sentiment Analysis — relocated here, beside the filters.
        n_available = len(filtered_df)
        ctrl_col, run_col = st.columns([3, 1])

        with ctrl_col:
            if n_available == 0:
                st.warning("No headlines match the current filters — adjust filters above to enable analysis.")
                sample_cap = 0
            elif n_available <= 20:
                st.caption(f"Only {n_available} headline(s) match filters — analyzing all of them.")
                sample_cap = n_available
            else:
                slider_max = min(500, n_available)
                default_val = min(150, slider_max)
                sample_cap = st.slider(
                    "Max headlines to analyze (controls API cost)", 20, slider_max, default_val
                )

        with run_col:
            st.write("")
            if n_available == 0:
                run_analysis = False
                st.button("Run Sentiment Analysis", type="primary", disabled=True)
            else:
                run_analysis = st.button("Run Sentiment Analysis", type="primary")

if "analyzed_df" not in st.session_state:
    st.session_state.analyzed_df = None
if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = analysis_mode

if run_analysis:
    if filtered_df.empty:
        st.warning("No headlines match the current filters.")
    else:
        with st.spinner("Calling Hugging Face API to analyze headlines..."):
            work_df = filtered_df.sample(
                min(sample_cap, len(filtered_df)), random_state=1
            ).reset_index(drop=True)
            try:
                results = classify_sentiment_batch(
                    work_df["headline"].tolist(), mode=analysis_mode
                )
            except GenAIUnavailableError as e:
                st.error(str(e))
                st.info(
                    "Tip: check that a valid Hugging Face API token is configured "
                    "in .streamlit/secrets.toml before running analysis."
                )
                st.stop()
            work_df["sentiment"] = [r.get("sentiment", "neutral") for r in results]
            work_df["keywords"] = [", ".join(r.get("keywords", [])) for r in results]
            st.session_state.analyzed_df = work_df
            st.session_state.analysis_mode = analysis_mode
        st.success(f"Analyzed {len(work_df):,} headlines using GenAI mode.")

# ---------------------------------------------------------------------------
# Sidebar: chatbot only
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("💬 Ask the Data")
    st.caption(
        "Ask things like 'What negative themes show up most?' or "
        "'Which category gets the most coverage?'"
    )
    context_df = st.session_state.analyzed_df if st.session_state.analyzed_df is not None else filtered_df
    chat_mode = st.session_state.get("analysis_mode", analysis_mode)
    user_question = st.text_input("Your question")
    ask_clicked = st.button("Ask")
    if ask_clicked and user_question.strip():
        with st.spinner("Thinking..."):
            try:
                answer = answer_question_about_data(user_question, context_df, mode=chat_mode)
            except GenAIUnavailableError as e:
                st.error(str(e))
                st.info(
                    "Tip: check that a valid Hugging Face API token is configured "
                    "in .streamlit/secrets.toml before asking questions."
                )
                st.stop()
        st.markdown(f"**Answer:** {answer}")

# ---------------------------------------------------------------------------
# At-a-glance status row
# ---------------------------------------------------------------------------
analyzed_count = 0 if st.session_state.analyzed_df is None else len(st.session_state.analyzed_df)
m1, m2, m3 = st.columns(3)
m1.metric("Total headlines", f"{len(df):,}")
m2.metric("Matching filters", f"{len(filtered_df):,}")
m3.metric("Analyzed", f"{analyzed_count:,}")

st.divider()

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
tab_overview, tab_viz = st.tabs(["📄 Data Preview", "📊 Visualizations"])

with tab_overview:
    st.subheader("Headline Flashcards")
    preview_df = filtered_df.head(50)
    st.caption(f"Showing {len(preview_df):,} of {len(filtered_df):,} matching headlines — one card per line.")

    if preview_df.empty:
        st.info("No headlines match the current filters.")

    for _, row in preview_df.iterrows():
        with st.container(border=True):
            date_txt = row["date"].strftime("%b %d, %Y") if pd.notna(row["date"]) else "No date"
            st.caption(f"🏷️ {_md_safe(row['category'])}  ·  📅 {date_txt}")
            st.markdown(f"#### {_md_safe(row['headline'])}")
            if str(row["short_description"]).strip():
                st.write(_md_safe(row["short_description"]))
            st.caption(f"✍️ {_md_safe(row['authors'])}")

with tab_viz:
    if st.session_state.analyzed_df is None:
        st.info("Run sentiment analysis above to see visualizations here.")
    else:
        adf = st.session_state.analyzed_df
        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                st.subheader("Sentiment Distribution")
                counts = adf["sentiment"].value_counts().reset_index()
                counts.columns = ["sentiment", "count"]
                fig = px.bar(
                    counts, x="sentiment", y="count", color="sentiment",
                    color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"},
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            with st.container(border=True):
                st.subheader("Sentiment by Category")
                by_cat = adf.groupby(["category", "sentiment"]).size().reset_index(name="count")
                fig2 = px.bar(
                    by_cat, x="category", y="count", color="sentiment", barmode="stack",
                    color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"},
                )
                fig2.update_layout(xaxis_title="", xaxis_tickangle=-30)
                st.plotly_chart(fig2, use_container_width=True)

        if adf["date"].notna().any():
            with st.container(border=True):
                st.subheader("Sentiment Trend Over Time")
                trend = adf.dropna(subset=["date"]).copy()
                trend["month"] = trend["date"].dt.to_period("M").astype(str)
                trend_counts = trend.groupby(["month", "sentiment"]).size().reset_index(name="count")
                fig3 = px.line(
                    trend_counts, x="month", y="count", color="sentiment", markers=True,
                    color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"},
                )
                st.plotly_chart(fig3, use_container_width=True)

        with st.container(border=True):
            st.subheader("Top Extracted Keywords")
            kw_series = adf["keywords"].str.split(", ").explode().dropna()
            kw_series = kw_series[kw_series != ""]
            top_kw = kw_series.value_counts().head(15).reset_index()
            top_kw.columns = ["keyword", "count"]
            fig4 = px.bar(top_kw, x="count", y="keyword", orientation="h")
            fig4.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig4, use_container_width=True)

        with st.expander("See analyzed headlines with sentiment + keywords"):
            st.dataframe(
                adf[["category", "headline", "sentiment", "keywords"]],
                use_container_width=True,
            )

st.divider()
st.caption(
    "Headlyn — built with Streamlit + Hugging Face for a GenAI dataset-analysis assignment."
)
