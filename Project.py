import os
import sys
import logging
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_PATH = "bank_transactions_data_2.csv"


# =========================
# 1. LOAD DATA
# =========================
log.info("Loading dataset...")
try:
    df = pd.read_csv(DATA_PATH, parse_dates=["TransactionDate", "PreviousTransactionDate"])
    log.info(f"Loaded {len(df)} rows from {DATA_PATH}")
except FileNotFoundError:
    log.error(f"Data file not found: {DATA_PATH}")
    sys.exit(1)
except Exception as e:
    log.error(f"Failed to load data: {e}")
    sys.exit(1)


# =========================
# 2. DATA QUALITY CHECKS
# =========================
from quality_checks.data_quality_engine import DataQualityEngine

log.info("Running data quality checks...")
try:
    dq_engine = DataQualityEngine(df)
    dq_results = dq_engine.run_all_checks()
    log.info(f"Data quality checks complete — {len(dq_results)} rules evaluated")
    print("\n=== DATA QUALITY RESULTS ===")
    print(dq_results.to_string(index=False))
except Exception as e:
    log.error(f"Data quality checks failed: {e}")
    sys.exit(1)


# =========================
# 3. ANOMALY DETECTION
# =========================
from anomaly_detection.anomaly_engine import AnomalyEngine

log.info("Running statistical anomaly detection...")
try:
    anomaly_engine = AnomalyEngine(df)
    amount_anomalies = anomaly_engine.detect_amount_zscore_anomalies()
    log.info(f"Amount Z-score anomalies: {len(amount_anomalies)}")

    account_amount_anomalies = anomaly_engine.detect_account_level_amount_anomalies()
    log.info(f"Account-level amount anomalies: {len(account_amount_anomalies)}")

    device_anomalies = anomaly_engine.detect_device_churn_anomalies()
    log.info(f"Device churn anomalies: {len(device_anomalies)}")

    ip_anomalies = anomaly_engine.detect_ip_churn_anomalies()
    log.info(f"IP churn anomalies: {len(ip_anomalies)}")
except Exception as e:
    log.error(f"Anomaly detection failed: {e}")
    sys.exit(1)


# =========================
# 4. UNIFIED ANOMALY TABLE
# =========================
log.info("Building unified anomaly table...")
try:
    unified_anomalies = anomaly_engine.build_unified_anomaly_table(
        amount_anomalies,
        account_amount_anomalies,
        device_anomalies,
        ip_anomalies,
    )
    log.info(f"Total anomalies consolidated: {len(unified_anomalies)}")
    print("\n=== UNIFIED ANOMALY TABLE (SAMPLE) ===")
    print(unified_anomalies.head(10).to_string(index=False))

    os.makedirs("outputs", exist_ok=True)
    unified_anomalies.to_csv("outputs/unified_anomalies.csv", index=False)
except Exception as e:
    log.error(f"Failed to build unified anomaly table: {e}")
    sys.exit(1)


# =========================
# 5. GENAI EXPLANATIONS (LANGCHAIN)
# =========================
from genai_explanations.langchain_explainer import LangChainExplanationEngine

api_key_present = bool(os.getenv("OPENAI_API_KEY"))
log.info(f"Generating GenAI explanations (live LLM: {api_key_present})...")

try:
    langchain_engine = LangChainExplanationEngine(
        model_name="gpt-4o-mini",
        temperature=0.3,
    )
    genai_df = langchain_engine.generate_explanations(unified_anomalies)
    genai_df.to_csv("outputs/genai_explanations.csv", index=False)
    log.info(f"Explanations generated for {len(genai_df)} anomalies")
    print("\n=== GENAI EXPLANATIONS (SAMPLE) ===")
    print(genai_df.head(5).to_string(index=False))
except Exception as e:
    log.error(f"GenAI explanation step failed: {e}")
    sys.exit(1)


# =========================
# 6. AGENTIC AI (LANGGRAPH)
# =========================
from agentic.risk_aggregator import RiskAggregator
from agentic.langgraph_agents import build_agent_graph

log.info("Aggregating risk scores...")
try:
    aggregator = RiskAggregator()
    risk_table = aggregator.aggregate(unified_anomalies)
    log.info(f"Risk table built: {len(risk_table)} entities scored")
    print("\n=== RISK TABLE (SAMPLE) ===")
    print(risk_table.head().to_string(index=False))
except Exception as e:
    log.error(f"Risk aggregation failed: {e}")
    sys.exit(1)

log.info("Running LangGraph agents...")
try:
    graph = build_agent_graph()
    decisions = []
    for _, row in risk_table.iterrows():
        result = graph.invoke(
            {
                "entity_id": row["EntityID"],
                "risk_level": row["risk_level"],
            }
        )
        decisions.append(result)

    decisions_df = pd.DataFrame(decisions)
    decisions_df.to_csv("outputs/agent_decisions.csv", index=False)
    log.info(f"Agent decisions written for {len(decisions_df)} entities")
    print("\n=== AGENT DECISIONS (SAMPLE) ===")
    print(decisions_df.head().to_string(index=False))
except Exception as e:
    log.error(f"LangGraph agent step failed: {e}")
    sys.exit(1)

log.info("Pipeline execution complete.")
