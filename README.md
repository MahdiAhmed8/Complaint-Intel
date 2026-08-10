# Sawt — Tunisian Complaint Intelligence

An inspectable Arabic/French NLP baseline for Tunisian companies. Sawt detects language, classifies complaint topic, sentiment and urgency, finds recurring operational problems, routes work to departments, and produces a short management brief.

The baseline deliberately uses word + character TF‑IDF and logistic regression. It makes no external AI/API calls.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app starts with a balanced synthetic bilingual dataset so every feature is immediately usable. Upload a CSV from the sidebar to classify real data. The only required column is `text`; optional columns are `date`, `product`, and `location`. If `topic`, `sentiment`, `urgency`, and `language` are present, they are treated as existing labels.

## Train from labelled company data

The CSV must contain `text`, `topic`, `sentiment`, and `urgency`. Use the six topic names from `complaint_intelligence/constants.py` and consistent sentiment/urgency labels.

```powershell
python train.py --data data/company_complaints.csv --output models/baseline.joblib
```

Training prints held-out accuracy and macro F1. The Streamlit app trains its cached demo model in memory; `train.py` is the reproducible persistence path for deployment.

## Evaluation approach

- Fixed 75/25 stratified train/test split (stratified by topic)
- Macro F1 and accuracy for every task
- Per-class report and confusion matrix
- Balanced logistic regression to reduce majority-class bias
- Confidence below 55% triggers a human-review notice

For a production evaluation, replace the synthetic data with reviewed historical complaints, deduplicate near-identical messages before splitting, and preferably use a time-based holdout. Monitor macro F1 by language, product, and location; tune the confidence threshold from actual review costs.

## Project structure

```text
app.py                              Streamlit dashboard and intake
train.py                            Reproducible training CLI
complaint_intelligence/modeling.py  TF-IDF + logistic regression pipeline
complaint_intelligence/language.py  Arabic/French script detection
complaint_intelligence/analytics.py Recurrence, keywords, management brief
complaint_intelligence/routing.py   Auditable department rules
complaint_intelligence/sample_data.py Bilingual demo data generator
tests/test_core.py                  Core behavior tests
```

## Production next steps

1. Label a representative sample with written annotation rules and measure annotator agreement.
2. Add Tunisian Arabizi handling (Arabic written in Latin characters) once examples exist.
3. Store predictions, confidence, corrections, ownership, and resolution time in a database.
4. Retrain on corrected cases and compare against this frozen baseline before considering a language model.
5. Add authentication and remove/redact personal data before deployment.
