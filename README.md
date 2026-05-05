# GenAI Bank Transaction Anomaly Detection

> End-to-end pipeline that detects suspicious bank transactions using statistical anomaly detection, LLM-generated explanations, and autonomous AI agent routing — all visualized in an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.2-green?logo=chainlink&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red?logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Flash-blue?logo=google&logoColor=white)

---

## Architecture

```
CSV Upload (Bank Transactions)
        │
        ▼
┌─────────────────────┐
│  Data Quality Engine │  ← 8 rule-based checks (integrity, financial, temporal, behavioral)
└─────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Anomaly Detection Engine│  ← Z-score, Account-level outlier, Device churn, IP churn
└─────────────────────────┘
        │
        ▼
┌──────────────────────┐
│  Unified Anomaly Table│  ← Consolidated with severity (HIGH / MEDIUM / LOW)
└──────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  GenAI Explanation Engine │  ← LangChain + Google Gemini 2.5 Flash
└──────────────────────────┘
        │
        ▼
┌──────────────────┐
│  Risk Aggregator  │  ← Severity-weighted scoring per entity
└──────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  LangGraph Agent Router                       │
│   LOW  → "Monitor periodically"              │
│   MED  → "Flag for analyst review"           │
│   HIGH → "Trigger immediate investigation"   │
└──────────────────────────────────────────────┘
        │
        ▼
  Streamlit Dashboard  +  CSV Outputs
```

---

## Key Features

- **8 data quality rules** across integrity, financial, temporal, and behavioral categories
- **4 statistical anomaly detectors**: global Z-score, account-level outlier, device churn, IP churn
- **Real AI explanations** via Google Gemini 2.5 Flash (LangChain) with mock fallback
- **LangGraph agent routing** — autonomous decision-making across 3 risk tiers
- **Interactive Streamlit dashboard** with 5 tabs, charts, filters, and CSV downloads
- **Rate-limited LLM calls** with live progress bar and daily quota handling
- **18 unit tests** covering all core modules

---

## How to Run

**1. Clone and install dependencies**
```bash
git clone <repo-url>
cd genai-fraud-detection
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**2. Add your Gemini API key**
```bash
cp .env.example .env
# Edit .env and paste your GOOGLE_API_KEY from aistudio.google.com
```

**3. Launch the dashboard**
```bash
streamlit run app.py
```

**4. Upload & run**
- Open `http://localhost:8501`
- Upload `bank_transactions_data_2.csv` (or any transaction CSV)
- Click **Run Pipeline** in the sidebar

**5. (Optional) Run tests**
```bash
python -m pytest tests/ -v
```

---

## Project Structure

```
genai-fraud-detection/
│
├── app.py                          # Streamlit dashboard (5 tabs)
├── Project.py                      # CLI pipeline runner
├── requirements.txt
├── .env.example                    # API key template
│
├── quality_checks/
│   └── data_quality_engine.py      # 8 validation rules
│
├── anomaly_detection/
│   └── anomaly_engine.py           # Z-score, account, device, IP detection
│
├── genai_explanations/
│   ├── langchain_explainer.py      # LangChain + Gemini (primary)
│   └── explanation_engine.py       # Standalone GenAI engine (alternative)
│
├── agentic/
│   ├── risk_aggregator.py          # Severity-weighted risk scoring
│   └── langgraph_agents.py         # LangGraph state graph + 3 agent paths
│
├── tests/
│   ├── test_data_quality.py        # 7 tests
│   ├── test_anomaly_engine.py      # 5 tests
│   └── test_risk_aggregator.py     # 6 tests
│
└── outputs/                        # Generated CSVs (git-ignored)
    ├── unified_anomalies.csv
    ├── genai_explanations.csv
    └── agent_decisions.csv
```

---

## Screenshots

### Dashboard Overview
> *Upload a transaction CSV, click Run Pipeline — the dashboard populates all 5 tabs automatically.*

![Overview Tab](docs/screenshots/overview.png)

### AI Explanations (Gemini)
> *Each HIGH and MEDIUM anomaly gets a real Gemini-generated analyst explanation.*

![AI Explanations Tab](docs/screenshots/ai_explanations.png)

### Risk & Agent Decisions
> *LangGraph routes each entity to a LOW / MEDIUM / HIGH agent with a specific action.*

![Risk Tab](docs/screenshots/risk_decisions.png)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data processing | pandas, numpy |
| LLM integration | LangChain, langchain-google-genai |
| AI model | Google Gemini 2.5 Flash (free tier) |
| Agent orchestration | LangGraph |
| Dashboard | Streamlit, Plotly |
| Testing | pytest |

---

## Notes

- Gemini free tier allows **20 requests/day** — the pipeline generates real AI explanations for the top 20 HIGH-priority anomalies and uses template explanations for the rest.
- No database required — runs entirely on CSV input.
- All outputs are saved to `outputs/` as CSV files for further analysis.
