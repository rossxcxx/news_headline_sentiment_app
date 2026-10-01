# 📰 News Headline Sentiment & Trend Explorer

Headlyn is a Streamlit web app that helps you understand the tone of the news. It ships with a dataset of news headlines that you can filter by category, author, and publication date, and browse as a stack of headline flashcards.

After filtering, Headlyn sends a sample of headlines to a Hugging Face sentiment model, which labels each one as positive, negative, or neutral. Recurring keywords are extracted locally using word-frequency analysis. The results appear as interactive charts showing sentiment distribution, sentiment by category, sentiment over time, and the most common keywords.


## 📊 Dataset source

This app is built around the **Kaggle "News Category Dataset"** by Rishabh
Misra — ~210,000 real news headlines from HuffPost (2012–2022) with category,
short description, author(s), date, and link metadata:

- **Dataset page:** https://www.kaggle.com/datasets/rmisra/news-category-dataset


## How it works

1. **Filter** — Widgets let you filter by category, author, and
   publication date range before running analysis (keeps things fast and
   light on the free tier).
2. **GenAI analysis** — `src/genai_analysis.py` sends each headline to the
   Hugging Face Inference API using the `cardiffnlp/twitter-roberta-base-sentiment-latest`
   model for sentiment/tone (positive/negative/neutral), and extracts
   keywords locally via simple word-frequency analysis. Results are cached
   with `st.cache_data` so re-running the app doesn't re-call the API on the
   same input. A **chat model** (`meta-llama/Llama-3.1-8B-Instruct` by
   default) powers the "Ask the Data" tab.
3. **Visualize** — Plotly charts show sentiment distribution, sentiment by
   category, sentiment trend over time, and top extracted keywords.
4. **Ask the Data** — a chatbot tab feeds a summary + sample of the
   currently filtered headlines to the model so you can ask free-form
   questions ("Which category gets the most coverage?").

## Streamlit App Link

https://newsheadlinesentimentapp-dmnejfmnnqxx3wpqsdebf2.streamlit.app
