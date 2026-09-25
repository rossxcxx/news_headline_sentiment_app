"""
data_loader.py
Loads and cleans the news headlines dataset.
"""

import pandas as pd
import streamlit as st


REQUIRED_COLUMNS = ["headline"]


@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    """
    Load a CSV of news headlines into a cleaned DataFrame.

    Expected (flexible) columns:
        headline_id, category, headline, short_description, authors,
        date, link

    The loader is forgiving: it will still work if some optional
    columns are missing, but 'headline' must be present.
    """
    df = pd.read_csv(file)

    # Normalize column names (strip spaces, lowercase, underscores)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Drop rows with empty headline text
    df = df.dropna(subset=["headline"])
    df["headline"] = df["headline"].astype(str).str.strip()
    df = df[df["headline"].str.len() > 0]

    # Deduplicate identical headlines
    df = df.drop_duplicates(subset=["headline"])

    # Fill in optional columns if missing, so the rest of the app can rely on them
    if "category" not in df.columns:
        df["category"] = "Uncategorized"
    if "short_description" not in df.columns:
        df["short_description"] = ""
    if "authors" not in df.columns:
        df["authors"] = "Unknown"
    if "link" not in df.columns:
        df["link"] = ""

    # Parse dates if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT

    df = df.reset_index(drop=True)
    return df


def get_filtered_data(
    df: pd.DataFrame,
    categories: list | None = None,
    authors: list | None = None,
    date_range: tuple | None = None,
) -> pd.DataFrame:
    """Apply sidebar filters to the cleaned dataframe."""
    filtered = df.copy()

    if categories:
        filtered = filtered[filtered["category"].isin(categories)]

    if authors:
        filtered = filtered[filtered["authors"].isin(authors)]

    if date_range and filtered["date"].notna().any():
        start, end = date_range
        filtered = filtered[
            (filtered["date"] >= pd.Timestamp(start))
            & (filtered["date"] <= pd.Timestamp(end))
        ]

    return filtered
