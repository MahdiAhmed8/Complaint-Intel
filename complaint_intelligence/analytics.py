"""Recurring-problem detection and concise management reporting."""

from __future__ import annotations

import pandas as pd
import spacy
from nltk import FreqDist

from .text import normalize_text

STOPWORDS = {
    "fr": {"le", "la", "les", "un", "une", "de", "des", "du", "et", "est", "je", "ma", "mon", "mes", "ce", "cette", "pour", "pas", "plus", "en", "a", "au", "que", "cela", "merci"},
    "ar": {"من", "في", "على", "ما", "هذا", "هذه", "انا", "الى", "عن", "مع", "كان", "فما", "موش", "برشة", "يرجى"},
}


def top_keywords(texts: pd.Series, language: str, limit: int = 8) -> list[tuple[str, int]]:
    """Extract frequent content tokens with spaCy and NLTK, without model downloads."""
    nlp = spacy.blank("fr" if language == "fr" else "xx")
    words: list[str] = []
    for text in texts.dropna().astype(str):
        doc = nlp.make_doc(normalize_text(text))
        words.extend(
            token.text for token in doc
            if token.is_alpha and len(token.text) > 2 and token.text not in STOPWORDS.get(language, set())
        )
    return FreqDist(words).most_common(limit)


def recurring_problems(df: pd.DataFrame, min_count: int = 2) -> pd.DataFrame:
    """Find recurring topic/product/location combinations, weighted by severity."""
    if df.empty:
        return pd.DataFrame(columns=["topic", "product", "location", "complaints", "high_urgency", "negative_rate", "score"])
    work = df.copy()
    for col in ("topic", "product", "location"):
        if col not in work:
            work[col] = "Unknown"
        work[col] = work[col].fillna("Unknown")
    urgency = work["urgency"] if "urgency" in work else pd.Series("medium", index=work.index)
    sentiment = work["sentiment"] if "sentiment" in work else pd.Series("neutral", index=work.index)
    work["is_high"] = urgency.eq("high").astype(int)
    work["is_negative"] = sentiment.eq("negative").astype(int)
    grouped = (
        work.groupby(["topic", "product", "location"], dropna=False)
        .agg(complaints=("text", "size"), high_urgency=("is_high", "sum"), negative_rate=("is_negative", "mean"))
        .reset_index()
    )
    grouped["score"] = grouped["complaints"] + 2 * grouped["high_urgency"] + grouped["negative_rate"]
    return grouped[grouped["complaints"] >= min_count].sort_values(["score", "complaints"], ascending=False)


def management_summary(df: pd.DataFrame) -> str:
    """Generate a short extractive/template summary with no external AI call."""
    if df.empty:
        return "No complaints match the current filters."
    total = len(df)
    topic_counts = df["topic"].value_counts()
    top_topic, top_count = topic_counts.index[0], int(topic_counts.iloc[0])
    negative = int(df["sentiment"].eq("negative").sum())
    urgent = int(df["urgency"].eq("high").sum())
    location = df["location"].mode().iloc[0] if "location" in df and not df["location"].dropna().empty else "unknown"
    department = df["department"].str.replace(" — Priority queue", "", regex=False).mode().iloc[0]
    trend = ""
    if "date" in df:
        dated = df.copy()
        dated["date"] = pd.to_datetime(dated["date"], errors="coerce")
        weekly = dated.dropna(subset=["date"]).set_index("date").resample("7D").size()
        if len(weekly) >= 2 and weekly.iloc[-2] > 0:
            change = (weekly.iloc[-1] - weekly.iloc[-2]) / weekly.iloc[-2] * 100
            trend = f" Volume in the latest 7-day period is {abs(change):.0f}% {'higher' if change >= 0 else 'lower'} than the previous period."
    return (
        f"{total} complaints were reviewed. {top_topic} is the leading issue ({top_count}, {top_count / total:.0%}), "
        f"concentrated most often in {location}. {negative / total:.0%} are negative and {urgent} require high-priority handling. "
        f"Management should ask {department} to investigate the dominant pattern first.{trend}"
    )
