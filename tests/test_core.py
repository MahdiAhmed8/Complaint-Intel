import pandas as pd

from complaint_intelligence.analytics import management_summary, recurring_problems
from complaint_intelligence.language import detect_language
from complaint_intelligence.modeling import train_models
from complaint_intelligence.routing import route_department
from complaint_intelligence.sample_data import generate_sample_data


def test_language_detection():
    assert detect_language("الطلبية متاعي ما وصلتش") == "ar"
    assert detect_language("Ma commande est en retard") == "fr"
    assert detect_language("123 !!!") == "unknown"


def test_routing_priority():
    assert route_department("Payment") == "Finance"
    assert "Priority queue" in route_department("Technical problem", "high")


def test_training_and_prediction():
    bundle = train_models(generate_sample_data())
    result = bundle.predict("Ma carte a été débitée deux fois. C'est urgent !")
    assert result["topic"] == "Payment"
    assert result["language"] == "fr"
    assert 0 <= result["topic_confidence"] <= 1
    arabic = bundle.predict("المنتوج وصل مكسّر ولازم حل توّا")
    assert arabic["language"] == "ar"
    assert arabic["topic"] == "Damaged product"


def test_recurring_and_summary():
    df = generate_sample_data().head(30)
    df["department"] = [route_department(t, u) for t, u in zip(df.topic, df.urgency)]
    recurring = recurring_problems(df)
    assert isinstance(recurring, pd.DataFrame)
    assert "complaints" in recurring.columns
    assert "complaints were reviewed" in management_summary(df)
