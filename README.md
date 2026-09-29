# 📰 News Headline Sentiment & Trend Explorer

A GenAI-powered Streamlit app that analyzes news headlines for sentiment
(tone), keywords/themes, and trends — with a built-in chatbot for asking
questions about the dataset.


## 📊 Dataset source

This app is built around the **Kaggle "News Category Dataset"** by Rishabh
Misra — ~210,000 real news headlines from HuffPost (2012–2022) with category,
short description, author(s), date, and link metadata:

- **Dataset page:** https://www.kaggle.com/datasets/rmisra/news-category-dataset


## How it works

1. **Load & clean** — `src/data_loader.py` loads the CSV with Pandas, strips
   empty/duplicate headlines, and normalizes column names.
2. **Filter** — sidebar widgets let you filter by category, author, and
   publication date range before running analysis (keeps things fast and
   light on the free tier).
3. **GenAI analysis** — `src/genai_analysis.py` sends each headline to the
   Hugging Face Inference API using the `cardiffnlp/twitter-roberta-base-sentiment-latest`
   model for sentiment/tone (positive/negative/neutral), and extracts
   keywords locally via simple word-frequency analysis. Results are cached
   with `st.cache_data` so re-running the app doesn't re-call the API on the
   same input. A **chat model** (`meta-llama/Llama-3.1-8B-Instruct` by
   default) powers the "Ask the Data" tab.
4. **Visualize** — Plotly charts show sentiment distribution, sentiment by
   category, sentiment trend over time, and top extracted keywords.
5. **Ask the Data** — a chatbot tab feeds a summary + sample of the
   currently filtered headlines to the model so you can ask free-form
   questions ("Which category gets the most coverage?").

## Next Goals (per assignment)

- ✅ Filters by category and author are already included in the sidebar.
- ✅ A basic "Ask the Data" chatbot tab is included.
- Ideas to extend further:
  - Add a category leaderboard ranking by average sentiment.
  - Cache GenAI results to disk/DB so re-deploys don't re-analyze from scratch.
  - Add support for OpenAI or Anthropic's Claude API as an alternative model provider.
  - Add headline-length or engagement weighting to sentiment aggregation.
