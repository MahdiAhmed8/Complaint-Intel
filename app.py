from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from complaint_intelligence.analytics import management_summary, recurring_problems, top_keywords
from complaint_intelligence.constants import TOPIC_COLORS
from complaint_intelligence.language import language_name
from complaint_intelligence.modeling import ModelBundle, train_models
from complaint_intelligence.routing import route_department
from complaint_intelligence.sample_data import generate_sample_data

st.set_page_config(page_title="Sawt | Complaint Intelligence", page_icon="◉", layout="wide")

st.markdown(
    """
    <style>
    :root { --ink:#17211d; --muted:#66736d; --green:#0d7c66; --cream:#f6f3eb; }
    .stApp { background: #f7f8f5; color: var(--ink); }
    [data-testid="stSidebar"] { background: #132c26; }
    [data-testid="stSidebar"] * { color: #eef5f1; }
    [data-testid="stSidebar"] .stMultiSelect span { color: #17211d; }
    .brand { font-family: Georgia, serif; font-size:2.15rem; letter-spacing:-.04em; font-weight:700; }
    .eyebrow { color:#0d7c66; font-weight:700; font-size:.73rem; letter-spacing:.15em; text-transform:uppercase; }
    .subtitle { color:#66736d; max-width:720px; margin-top:-.4rem; }
    .metric-card { padding:1.15rem 1.25rem; border:1px solid #e1e5df; border-radius:14px; background:white; min-height:126px; }
    .metric-label { color:#708079; font-size:.75rem; text-transform:uppercase; letter-spacing:.08em; font-weight:650; }
    .metric-value { color:#17211d; font-family:Georgia,serif; font-size:2rem; line-height:1.2; margin:.35rem 0; }
    .metric-note { color:#708079; font-size:.78rem; }
    .summary { background:#ecf5f0; border-left:4px solid #0d7c66; padding:1.1rem 1.25rem; border-radius:0 10px 10px 0; }
    .route { background:#172f29; color:white; border-radius:12px; padding:1rem 1.2rem; }
    div[data-testid="stDataFrame"] { border:1px solid #e1e5df; border-radius:10px; overflow:hidden; }
    .stTabs [data-baseweb="tab-list"] { gap:1.5rem; }
    .stTabs [data-baseweb="tab"] { padding-left:0; padding-right:0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def sample_data() -> pd.DataFrame:
    return generate_sample_data()


@st.cache_resource(show_spinner="Training transparent baseline models…")
def model_bundle() -> ModelBundle:
    return train_models(sample_data())


def enrich(df: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    out = df.copy()
    if "date" in out:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").fillna(pd.Timestamp.today())
    else:
        out["date"] = pd.Timestamp.today().normalize()
    for col, fallback in (("product", "Unspecified"), ("location", "Unspecified")):
        if col not in out:
            out[col] = fallback
        out[col] = out[col].fillna(fallback).astype(str)
    needed = {"topic", "sentiment", "urgency", "language"} - set(out.columns)
    if needed:
        predictions = out["text"].astype(str).map(bundle.predict).apply(pd.Series)
        for col in needed:
            out[col] = predictions[col]
        for target in ("topic", "sentiment", "urgency"):
            out[f"{target}_confidence"] = predictions[f"{target}_confidence"]
    out["department"] = [route_department(t, u) for t, u in zip(out["topic"], out["urgency"])]
    return out


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


bundle = model_bundle()
base_df = sample_data()

with st.sidebar:
    st.markdown("## ◉ Sawt")
    st.caption("Complaint intelligence / صوت الحريف")
    st.markdown("---")
    upload = st.file_uploader("Upload complaints", type=["csv"], help="CSV must contain a text column. Labels are optional.")
    if upload:
        try:
            uploaded = pd.read_csv(upload)
            if "text" not in uploaded:
                st.error("The CSV needs a ‘text’ column.")
                st.stop()
            data = enrich(uploaded, bundle)
            data_source = "Uploaded data"
        except Exception as exc:
            st.error(f"Could not read file: {exc}")
            st.stop()
    else:
        data = enrich(base_df, bundle)
        data_source = "Demonstration data"

    st.markdown("### Filters")
    products = st.multiselect("Product", sorted(data["product"].unique()), default=[])
    locations = st.multiselect("Location", sorted(data["location"].unique()), default=[])
    topics = st.multiselect("Topic", sorted(data["topic"].unique()), default=[])
    min_date, max_date = data["date"].min().date(), data["date"].max().date()
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    st.markdown("---")
    st.caption(f"{data_source} · {len(data):,} rows")
    st.download_button("Download classified CSV", data.to_csv(index=False).encode("utf-8-sig"), "classified_complaints.csv", "text/csv", use_container_width=True)

filtered = data.copy()
if products:
    filtered = filtered[filtered["product"].isin(products)]
if locations:
    filtered = filtered[filtered["location"].isin(locations)]
if topics:
    filtered = filtered[filtered["topic"].isin(topics)]
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    filtered = filtered[filtered["date"].dt.date.between(date_range[0], date_range[1])]

st.markdown('<div class="eyebrow">Bilingual operations intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="brand">The voice behind every complaint.</div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Arabic and French complaint triage built on an inspectable TF‑IDF + logistic regression baseline.</p>', unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Recurring problems", "Live triage", "Model quality", "Complaint log"])

with tabs[0]:
    if filtered.empty:
        st.info("No complaints match these filters.")
    else:
        total = len(filtered)
        negative = filtered["sentiment"].eq("negative").mean()
        urgent = int(filtered["urgency"].eq("high").sum())
        top_topic = filtered["topic"].mode().iloc[0]
        cols = st.columns(4)
        with cols[0]: metric_card("Complaints", f"{total:,}", "in the selected period")
        with cols[1]: metric_card("Negative", f"{negative:.0%}", "share of filtered complaints")
        with cols[2]: metric_card("High urgency", f"{urgent:,}", "sent to priority queues")
        with cols[3]: metric_card("Leading topic", top_topic, "most frequent classification")

        st.markdown("### Management brief")
        st.markdown(f'<div class="summary">{management_summary(filtered)}</div>', unsafe_allow_html=True)
        left, right = st.columns([1.25, 1])
        with left:
            st.markdown("### Complaint volume over time")
            daily = filtered.set_index("date").resample("W").size().rename("complaints").reset_index()
            fig = px.area(daily, x="date", y="complaints", markers=True, color_discrete_sequence=["#0d7c66"])
            fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), height=310, yaxis_title=None, xaxis_title=None, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown("### Topics")
            counts = filtered["topic"].value_counts().rename_axis("topic").reset_index(name="complaints")
            fig = px.bar(counts, y="topic", x="complaints", orientation="h", color="topic", color_discrete_map=TOPIC_COLORS)
            fig.update_layout(showlegend=False, margin=dict(l=0, r=10, t=10, b=0), height=310, yaxis_title=None, xaxis_title=None, plot_bgcolor="rgba(0,0,0,0)", yaxis={"categoryorder":"total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.markdown("### Product × sentiment")
            matrix = pd.crosstab(filtered["product"], filtered["sentiment"])
            st.dataframe(matrix, use_container_width=True)
        with right:
            st.markdown("### Location × urgency")
            matrix = pd.crosstab(filtered["location"], filtered["urgency"])
            st.dataframe(matrix, use_container_width=True)

with tabs[1]:
    st.markdown("### Patterns worth investigating")
    st.caption("A recurrence combines topic, product and location. The score weights high-urgency and negative complaints more heavily.")
    recurring = recurring_problems(filtered, min_count=2)
    if recurring.empty:
        st.info("No repeated patterns meet the minimum of two complaints.")
    else:
        st.dataframe(
            recurring,
            use_container_width=True,
            hide_index=True,
            column_config={"negative_rate": st.column_config.ProgressColumn("Negative rate", min_value=0, max_value=1, format="%.0%%"), "score": st.column_config.NumberColumn("Priority score", format="%.1f")},
        )
    st.markdown("### Frequent language-specific terms")
    kw_cols = st.columns(2)
    for column, language in zip(kw_cols, ("ar", "fr")):
        with column:
            terms = top_keywords(filtered.loc[filtered["language"].eq(language), "text"], language)
            st.markdown(f"**{language_name(language)}**")
            st.write(" · ".join(f"{word} ({count})" for word, count in terms) or "No terms available")

with tabs[2]:
    st.markdown("### Classify and route a new complaint")
    with st.form("triage"):
        complaint = st.text_area("Complaint text", height=150, placeholder="الطلبية ما وصلتش... / Ma commande n'est pas arrivée…")
        c1, c2 = st.columns(2)
        product = c1.selectbox("Product", sorted(data["product"].unique()))
        location = c2.selectbox("Location", sorted(data["location"].unique()))
        submitted = st.form_submit_button("Analyse complaint", type="primary", use_container_width=True)
    if submitted:
        if not complaint.strip():
            st.warning("Enter a complaint first.")
        else:
            result = bundle.predict(complaint)
            st.markdown(f'<div class="route"><small>ROUTE TO</small><br><b>{result["department"]}</b></div>', unsafe_allow_html=True)
            result_cols = st.columns(4)
            result_cols[0].metric("Language", language_name(result["language"]))
            result_cols[1].metric("Topic", result["topic"], f'{result["topic_confidence"]:.0%} confidence')
            result_cols[2].metric("Sentiment", result["sentiment"], f'{result["sentiment_confidence"]:.0%} confidence')
            result_cols[3].metric("Urgency", result["urgency"], f'{result["urgency_confidence"]:.0%} confidence')
            low = min(result[f"{key}_confidence"] for key in ("topic", "sentiment", "urgency")) < 0.55
            if low:
                st.warning("At least one prediction is low confidence. Send it for human review before acting.")

with tabs[3]:
    st.markdown("### Held-out baseline evaluation")
    st.warning("These scores use synthetic demonstration data and validate the pipeline—not real-world business performance. Replace the data and retrain before production use.")
    cards = st.columns(3)
    for card, target in zip(cards, ("topic", "sentiment", "urgency")):
        score = bundle.metrics[target]
        with card:
            metric_card(target.title(), f'{score["macro_f1"]:.1%}', f'Macro F1 · accuracy {score["accuracy"]:.1%} · n={score["test_rows"]}')
    target = st.selectbox("Inspect confusion matrix", ["topic", "sentiment", "urgency"])
    metric = bundle.metrics[target]
    cm = pd.DataFrame(metric["confusion_matrix"], index=metric["labels"], columns=metric["labels"])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale=[[0, "#f2f5f2"], [1, "#0d7c66"]], labels=dict(x="Predicted", y="Actual", color="Count"))
    fig.update_layout(height=440, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("What this model does"):
        st.markdown("Each task has its own logistic-regression classifier. Word (1–2 gram) and character (3–5 gram) TF‑IDF features are combined; character features help with dialect, spelling variation and Arabic/French mixing. Training uses a fixed stratified split and class balancing. Language detection is a Unicode-script rule, and routing is a visible topic-to-department map.")

with tabs[4]:
    st.markdown("### Classified complaint log")
    show_cols = [c for c in ["complaint_id", "date", "language", "text", "topic", "sentiment", "urgency", "product", "location", "department"] if c in filtered]
    st.dataframe(filtered[show_cols].sort_values("date", ascending=False), use_container_width=True, hide_index=True, height=520)
