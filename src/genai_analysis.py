"""
genai_analysis.py
Handles sentiment/tone classification, keyword extraction, and Q&A about the
news headline dataset. Supports two modes:

  - "genai"   : calls the Hugging Face Inference API (free tier, token-based)
  - "offline" : a free, local, lexicon-based fallback with no API calls at
                all, so the app is fully testable/demoable with zero setup.

Hugging Face setup:
  1. Create a token at https://huggingface.co/settings/tokens (a "Read" token
     with "Inference" permission is enough for the free Inference Providers API).
  2. Add it to `.streamlit/secrets.toml` as HF_TOKEN = "hf_..." (locally) or
     in Streamlit Community Cloud's Secrets settings (when deployed).

Models used (swap these constants if a model becomes unavailable on your
Hugging Face plan/provider):
  - SENTIMENT_MODEL: a dedicated 3-class sentiment classifier
  - CHAT_MODEL: an instruction-tuned chat model for the "Ask the Data" tab
"""

import re
import time
from collections import Counter
from typing import List, Dict

import pandas as pd
import streamlit as st
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
CHAT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SENTIMENT_PROVIDER = "hf-inference"  # this classifier is hosted on HF's own infra
CHAT_PROVIDER = "auto"  # let HF route to whichever partner (Cerebras, Together, etc.) hosts this model

_LABEL_MAP = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
}

# ---------------------------------------------------------------------------
# Hugging Face client
# ---------------------------------------------------------------------------

def get_client(provider: str = SENTIMENT_PROVIDER) -> InferenceClient:
    hf_token = st.secrets.get("HF_TOKEN", None)
    if not hf_token:
        st.error(
            "No HF_TOKEN found in Streamlit secrets. Create a token at "
            "https://huggingface.co/settings/tokens and add it to "
            "`.streamlit/secrets.toml` (locally) or Streamlit Community "
            "Cloud's Secrets settings, or switch to 'Offline (free demo)' "
            "mode in the sidebar."
        )
        st.stop()
    return InferenceClient(provider=provider, api_key=hf_token)


class GenAIUnavailableError(Exception):
    """Raised when the Hugging Face API call fails (auth, quota, model down, etc)."""


# ---------------------------------------------------------------------------
# GenAI (Hugging Face) mode
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def classify_sentiment_batch_genai(headlines: List[str]) -> List[Dict]:
    """
    Calls the Hugging Face Inference API's text-classification task once per
    headline using a dedicated sentiment model, and extracts simple keywords
    locally (Hugging Face's free-tier classifiers return only a label +
    score, not keywords, so keyword extraction is done with a lightweight
    local frequency method regardless of mode).
    """
    client = get_client(provider=SENTIMENT_PROVIDER)
    results = []

    for text in headlines:
        try:
            raw = client.text_classification(text, model=SENTIMENT_MODEL)
        except HfHubHTTPError as e:
            raise GenAIUnavailableError(
                "Hugging Face rejected the request. This usually means: the "
                "token is invalid/missing the 'Inference' permission, you've "
                "hit the free-tier rate limit, or the model is temporarily "
                "unavailable. Check your token at "
                "https://huggingface.co/settings/tokens, or switch to "
                "'Offline (free demo)' mode in the sidebar. "
                f"Details: {e}"
            ) from e
        except Exception as e:
            raise GenAIUnavailableError(f"Hugging Face API error: {e}") from e

        top_label = max(raw, key=lambda r: r["score"])["label"].lower()
        sentiment = _LABEL_MAP.get(top_label, "neutral")

        keywords = _extract_keywords_local(text)
        results.append({"sentiment": sentiment, "keywords": keywords})

        time.sleep(0.05)  # light courtesy delay for the free tier

    return results


def _top_keywords(df: pd.DataFrame, sentiment: str | None = None, n: int = 8) -> str:
    """Return 'word (count), word (count)...' for the most common extracted keywords."""
    if "keywords" not in df.columns:
        return "N/A"
    sub = df if sentiment is None else df[df["sentiment"] == sentiment]
    kw = sub["keywords"].str.split(", ").explode().dropna()
    kw = kw[kw != ""]
    if kw.empty:
        return "N/A"
    return ", ".join(f"{w} ({c})" for w, c in kw.value_counts().head(n).items())


def _build_chat_context(df: pd.DataFrame, full_df: pd.DataFrame | None = None) -> str:
    """
    Build an exact, pre-computed fact sheet for the chat model.

    All counting is done here in pandas so the model only has to explain the
    numbers, never compute them (LLMs are unreliable at counting).

    df       : the analyzed sample (has sentiment/keywords) if analysis was run,
               otherwise the filtered view.
    full_df  : the full filtered view, used for category coverage.
    """
    coverage_df = full_df if full_df is not None else df
    has_sentiment = "sentiment" in df.columns
    lines = []

    # --- Coverage (full filtered view) ---------------------------------
    total_view = len(coverage_df)
    lines.append(f"HEADLINES IN CURRENT FILTERED VIEW: {total_view}")
    if coverage_df["date"].notna().any():
        lines.append(
            f"Date range: {coverage_df['date'].min().date()} to {coverage_df['date'].max().date()}"
        )

    cat_counts = coverage_df["category"].value_counts()
    if not cat_counts.empty:
        lines.append("\nNEWS CATEGORY COVERAGE (number of headlines per news category, most to least):")
        for cat, n in cat_counts.head(10).items():
            lines.append(f"- {cat}: {n} ({n / total_view:.0%})")
        lines.append(f"Most covered news category: {cat_counts.index[0]} ({cat_counts.iloc[0]} headlines)")

    # --- Sentiment (analyzed sample only) ------------------------------
    if not has_sentiment:
        lines.append(
            "\nSENTIMENT: sentiment analysis has NOT been run yet, so there are no "
            "sentiment results. If asked about sentiment, tell the user to click "
            "'Run Sentiment Analysis' first."
        )
        sample = df.sample(min(25, len(df)), random_state=1)
        sample_lines = [f"- [{r.category}] {r.headline}" for r in sample.itertuples()]
    else:
        n_an = len(df)
        counts = df["sentiment"].value_counts()
        pos, neg, neu = (int(counts.get(k, 0)) for k in ("positive", "negative", "neutral"))
        lines.append(f"\nSENTIMENT RESULTS (analyzed sample of {n_an} headlines):")
        lines.append(
            f"- positive: {pos} ({pos / n_an:.0%}), negative: {neg} ({neg / n_an:.0%}), "
            f"neutral: {neu} ({neu / n_an:.0%})"
        )

        by_cat = df.groupby(["category", "sentiment"]).size().unstack(fill_value=0)
        for col in ("positive", "negative", "neutral"):
            if col not in by_cat.columns:
                by_cat[col] = 0
        by_cat["total"] = by_cat[["positive", "negative", "neutral"]].sum(axis=1)
        by_cat = by_cat.sort_values("total", ascending=False).head(8)
        lines.append("\nSENTIMENT BY NEWS CATEGORY (analyzed sample):")
        for cat, r in by_cat.iterrows():
            lines.append(
                f"- {cat}: {int(r['positive'])} positive, {int(r['negative'])} negative, "
                f"{int(r['neutral'])} neutral"
            )
        if by_cat["negative"].max() > 0:
            lines.append(
                f"Category with the most negative headlines: {by_cat['negative'].idxmax()} "
                f"({int(by_cat['negative'].max())})"
            )

        lines.append("\nTOP KEYWORDS (word frequency, with counts):")
        lines.append(f"- overall: {_top_keywords(df)}")
        lines.append(f"- in negative headlines: {_top_keywords(df, 'negative')}")
        lines.append(f"- in positive headlines: {_top_keywords(df, 'positive')}")

        if df["date"].notna().any():
            t = df.dropna(subset=["date"]).copy()
            t["month"] = t["date"].dt.to_period("M").astype(str)
            tm = t.groupby(["month", "sentiment"]).size().unstack(fill_value=0).tail(6)
            lines.append("\nRECENT MONTHLY SENTIMENT TREND (analyzed sample):")
            for month, r in tm.iterrows():
                lines.append(
                    f"- {month}: " + ", ".join(f"{k} {int(r.get(k, 0))}" for k in ("positive", "negative", "neutral"))
                )

        parts = []
        for label, k in (("negative", 12), ("positive", 8), ("neutral", 5)):
            sub = df[df["sentiment"] == label]
            parts.append(sub.sample(min(k, len(sub)), random_state=1))
        sample = pd.concat(parts)
        sample_lines = [f"- [{r.category} | {r.sentiment}] {r.headline}" for r in sample.itertuples()]

    lines.append("\nEXAMPLE HEADLINES ([news category | sentiment] headline):")
    lines.extend(sample_lines)
    return "\n".join(lines)


def answer_question_about_data_genai(
    question: str, df: pd.DataFrame, full_df: pd.DataFrame | None = None
) -> str:
    """RAG-lite chatbot: answers from a pre-computed fact sheet using a HF chat model."""
    client = get_client(provider=CHAT_PROVIDER)

    if len(df) == 0:
        return "There are no headlines in the current view to analyze."

    context = _build_chat_context(df, full_df)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a data analyst answering questions about a news headline "
                "dataset. You are given a FACT SHEET with exact, pre-computed numbers.\n"
                "Rules:\n"
                "1. Use the numbers in the fact sheet exactly as written. Never count, "
                "estimate or recalculate them yourself.\n"
                "2. 'Category' means a NEWS category such as SPORTS or POLITICS. "
                "Positive, negative and neutral are SENTIMENT labels, never categories.\n"
                "3. For questions about themes, use the keyword lists and the example "
                "headlines, and say which keywords or headlines support your answer.\n"
                "4. If the fact sheet does not contain what is needed, say so plainly "
                "instead of guessing.\n"
                "5. Be concise: a short direct answer first, then brief supporting detail."
            ),
        },
        {"role": "user", "content": f"FACT SHEET\n{context}\n\nQuestion: {question}"},
    ]

    try:
        response = client.chat_completion(
            messages, model=CHAT_MODEL, max_tokens=500, temperature=0.2
        )
    except HfHubHTTPError as e:
        raise GenAIUnavailableError(
            "Hugging Face rejected the chat request. Check your token's "
            "'Inference' permission and try again in a moment (free-tier rate "
            f"limit). Details: {e}"
        ) from e
    except Exception as e:
        raise GenAIUnavailableError(f"Hugging Face API error: {e}") from e

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Shared local keyword extraction (used in both genai and offline modes)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "is", "it", "this", "that", "and", "or", "but", "of",
    "to", "in", "on", "for", "with", "was", "were", "are", "i", "you", "my",
    "at", "as", "be", "if", "so", "not", "very", "really", "just", "its",
    "have", "has", "had", "can", "will", "would", "than", "then", "there",
    "their", "they", "them", "after", "about", "into", "over", "amid",
    "news", "week", "says", "say",
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _extract_keywords_local(text: str, top_n: int = 3) -> List[str]:
    tokens = [t for t in _tokenize(text) if t not in _STOPWORDS and len(t) > 3]
    return [w for w, _ in Counter(tokens).most_common(top_n)]


# ---------------------------------------------------------------------------
# Offline (free, no API) fallback mode
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = {
    "win", "wins", "milestone", "praised", "praise", "hope", "celebrates",
    "breakthrough", "impressed", "inspired", "exceeds", "boosting",
    "historic", "smooth", "confidence", "surprising", "bold", "impressive",
}
_NEGATIVE_WORDS = {
    "backlash", "struggles", "setback", "setbacks", "fire", "costly",
    "mistake", "disappoints", "underwhelming", "delays", "frustrated",
    "criticized", "outrage", "controversial", "suffers", "concerns",
    "chaotic", "avoidable", "lack",
}


def classify_sentiment_batch_offline(headlines: List[str]) -> List[Dict]:
    """
    Free, local, lexicon-based sentiment classifier + naive keyword extraction.
    No API calls, no token needed. Good enough for demoing the app's UI/UX,
    not a substitute for real model-quality analysis.
    """
    results = []
    for text in headlines:
        tokens = _tokenize(text)
        pos_hits = sum(1 for t in tokens if t in _POSITIVE_WORDS)
        neg_hits = sum(1 for t in tokens if t in _NEGATIVE_WORDS)

        if pos_hits > neg_hits:
            sentiment = "positive"
        elif neg_hits > pos_hits:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        results.append({"sentiment": sentiment, "keywords": _extract_keywords_local(text)})
    return results


def answer_question_about_data_offline(question: str, df: pd.DataFrame) -> str:
    """
    Free, local, rule-based 'chatbot' fallback: answers a small set of
    common question patterns using stats computed directly from the data,
    instead of calling any API.
    """
    total = len(df)
    if total == 0:
        return "There are no headlines in the current view to analyze."

    q = question.lower()
    has_sentiment = "sentiment" in df.columns

    if has_sentiment:
        counts = df["sentiment"].value_counts()
        pos = int(counts.get("positive", 0))
        neg = int(counts.get("negative", 0))
        neu = int(counts.get("neutral", 0))
    else:
        pos = neg = neu = None

    if any(w in q for w in ["negative", "problem", "issue", "bad", "worst"]):
        if "keywords" in df.columns:
            neg_kw = (
                df.loc[df.get("sentiment") == "negative", "keywords"]
                .str.split(", ")
                .explode()
                .dropna()
            )
            neg_kw = neg_kw[neg_kw != ""]
            if not neg_kw.empty:
                top = ", ".join(neg_kw.value_counts().head(5).index.tolist())
                return f"[Offline mode] Most common negative-headline themes: {top}."
        return "[Offline mode] Run sentiment analysis first so I can identify negative themes."

    if any(w in q for w in ["popular", "most covered", "top category", "which category"]):
        if "category" in df.columns:
            top_cats = df["category"].value_counts().head(3)
            listing = ", ".join(f"{c} ({n} headlines)" for c, n in top_cats.items())
            return f"[Offline mode] Categories with the most headlines in this view: {listing}."

    if any(w in q for w in ["trend", "over time", "improving", "declining"]):
        if "date" in df.columns and df["date"].notna().any() and has_sentiment:
            trend = df.dropna(subset=["date"]).copy()
            trend["month"] = trend["date"].dt.to_period("M")
            recent = trend[trend["month"] == trend["month"].max()]
            recent_pos_rate = (recent["sentiment"] == "positive").mean() if len(recent) else None
            if recent_pos_rate is not None:
                return (
                    f"[Offline mode] In the most recent month in this view, "
                    f"{recent_pos_rate:.0%} of headlines were positive "
                    f"(out of {pos + neg + neu} total analyzed)."
                )

    if pos is not None:
        return (
            f"[Offline mode - no API used] Of {total} headlines in the current view: "
            f"{pos} positive, {neg} negative, {neu} neutral. "
            f"For richer, natural-language answers, add an HF_TOKEN and "
            f"switch to GenAI mode in the sidebar."
        )

    return (
        "[Offline mode] Run sentiment analysis first (sidebar) so I have "
        "sentiment data to answer questions about."
    )


# ---------------------------------------------------------------------------
# Unified entry points used by app.py
# ---------------------------------------------------------------------------

def classify_sentiment_batch(headlines: List[str], mode: str = "genai") -> List[Dict]:
    if mode == "offline":
        return classify_sentiment_batch_offline(headlines)
    return classify_sentiment_batch_genai(headlines)


def answer_question_about_data(
    question: str,
    df: pd.DataFrame,
    mode: str = "genai",
    full_df: pd.DataFrame | None = None,
) -> str:
    if mode == "offline":
        return answer_question_about_data_offline(question, df)
    return answer_question_about_data_genai(question, df, full_df=full_df)
