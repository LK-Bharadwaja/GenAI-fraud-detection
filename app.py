import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(APP_DIR, ".env"))
except ImportError:
    pass

from quality_checks.data_quality_engine import DataQualityEngine
from anomaly_detection.anomaly_engine import AnomalyEngine
from genai_explanations.langchain_explainer import LangChainExplanationEngine
from agentic.risk_aggregator import RiskAggregator
from agentic.langgraph_agents import build_agent_graph

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GenAI Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEVERITY_COLORS = {"HIGH": "#FF4B4B", "MEDIUM": "#FFA500", "LOW": "#21C354"}
STATUS_COLORS   = {"PASS": "#21C354", "FAIL": "#FF4B4B", "FLAG": "#FFA500"}


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank.png", width=60)
    st.title("GenAI Fraud Detection")
    st.caption("Bank Transaction Anomaly Analysis")
    st.divider()

    uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

    st.subheader("Detection Thresholds")
    z_threshold      = st.slider("Z-Score Threshold",     2.0, 5.0, 3.0, 0.5,
                                  help="Higher = fewer but more extreme anomalies")
    device_threshold = st.slider("Device Churn Threshold", 3, 10, 5,
                                  help="Max unique devices before flagging an account")
    ip_threshold     = st.slider("IP Churn Threshold",     3, 10, 5,
                                  help="Max unique IPs before flagging an account")

    st.divider()
    if os.getenv("GOOGLE_API_KEY"):
        st.success("Live LLM: Gemini (Google)")
        llm_live = True
        llm_label = "Gemini (Google)"
    elif os.getenv("OPENAI_API_KEY"):
        st.success("Live LLM: OpenAI")
        llm_live = True
        llm_label = "OpenAI"
    else:
        st.warning("Mock LLM — add GOOGLE_API_KEY to .env")
        llm_live = False
        llm_label = "Mock"

    run_btn = st.button("Run Pipeline", type="primary", use_container_width=True,
                         disabled=uploaded_file is None)

    if uploaded_file is None:
        st.info("Upload a CSV to begin.")


# ─────────────────────────────────────────────
# Landing page (no file yet)
# ─────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("## GenAI Bank Transaction Anomaly Detection")
    st.markdown(
        "This pipeline combines **rule-based data quality checks**, "
        "**statistical anomaly detection**, **LLM-generated explanations**, "
        "and **LangGraph agent routing** to identify suspicious transactions."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Quality Rules", "8")
    col2.metric("Detection Methods", "4")
    col3.metric("Agent Paths", "3 (LOW / MEDIUM / HIGH)")
    col4.metric("LLM Model", llm_label)

    st.divider()
    st.markdown("### Pipeline Architecture")
    st.code(
        "CSV Upload  →  Data Quality (8 rules)\n"
        "           →  Anomaly Detection (Z-score, Account, Device, IP)\n"
        "           →  Unified Anomaly Table\n"
        "           →  GenAI Explanations (LangChain + OpenAI)\n"
        "           →  Risk Aggregation\n"
        "           →  LangGraph Agent Routing  →  Decisions",
        language="text",
    )
    st.stop()


# ─────────────────────────────────────────────
# Pipeline — base (cached, no LLM)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline_base(file_bytes, z_thr, dev_thr, ip_thr, run_id=0):
    import io
    df = pd.read_csv(
        io.BytesIO(file_bytes),
        parse_dates=["TransactionDate", "PreviousTransactionDate"],
    )
    dq_engine  = DataQualityEngine(df)
    dq_results = dq_engine.run_all_checks()

    ae           = AnomalyEngine(df)
    amount_anom  = ae.detect_amount_zscore_anomalies(z_threshold=z_thr)
    account_anom = ae.detect_account_level_amount_anomalies()
    device_anom  = ae.detect_device_churn_anomalies(device_threshold=dev_thr)
    ip_anom      = ae.detect_ip_churn_anomalies(ip_threshold=ip_thr)
    unified      = ae.build_unified_anomaly_table(amount_anom, account_anom, device_anom, ip_anom)

    risk_table = RiskAggregator().aggregate(unified)
    graph      = build_agent_graph()
    decisions  = [
        graph.invoke({"entity_id": row["EntityID"], "risk_level": row["risk_level"]})
        for _, row in risk_table.iterrows()
    ]
    decisions_df = pd.DataFrame(decisions)

    return df, dq_results, unified, risk_table, decisions_df


if run_btn:
    for key in ["pipeline_results", "genai_df"]:
        st.session_state.pop(key, None)
    st.cache_data.clear()
    st.session_state["run_id"] = st.session_state.get("run_id", 0) + 1

# ── Step 1: fast base pipeline ──
if "pipeline_results" not in st.session_state and run_btn:
    with st.spinner("Running data quality checks and anomaly detection…"):
        try:
            base = run_pipeline_base(
                uploaded_file.getvalue(),
                z_threshold, device_threshold, ip_threshold,
                run_id=st.session_state.get("run_id", 0),
            )
            st.session_state["pipeline_results"] = base
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

if "pipeline_results" not in st.session_state:
    st.info("Click **Run Pipeline** in the sidebar to start.")
    st.stop()

df, dq_results, unified, risk_table, decisions_df = st.session_state["pipeline_results"]

# ── Step 2: LLM explanations (live, with progress bar) ──
MAX_LLM_CALLS = 20   # matches gemini-2.5-flash free-tier daily quota
DELAY_SECONDS = 13.0  # 60s / 5 req-per-min = 12s min; 13s adds a safety buffer

if "genai_df" not in st.session_state and run_btn:
    hm_count  = int((unified["severity"].isin(["HIGH", "MEDIUM"])).sum())
    llm_count = min(hm_count, MAX_LLM_CALLS)
    est_min   = round(llm_count * DELAY_SECONDS / 60, 1)

    st.info(
        f"Generating Gemini explanations for the top **{llm_count}** HIGH/MEDIUM anomalies "
        f"(free tier: 5 req/min → ~{est_min} min). "
        f"Remaining {len(unified) - llm_count} anomalies will use template explanations."
    )
    prog_bar    = st.progress(0.0, text="Starting Gemini calls…")
    status_text = st.empty()

    def _on_progress(done, total):
        prog_bar.progress(done / total, text=f"Gemini: {done}/{total} explanations generated")
        status_text.text(f"Completed {done} of {total} — please wait…")

    try:
        engine   = LangChainExplanationEngine()
        genai_df = engine.generate_explanations(
            unified,
            severity_filter=["HIGH", "MEDIUM"],
            delay_seconds=DELAY_SECONDS,
            max_llm_calls=MAX_LLM_CALLS,
            progress_callback=_on_progress,
        )
        prog_bar.progress(1.0, text="Done!")
        status_text.empty()
        st.session_state["genai_df"] = genai_df
        gemini_n = int((genai_df["llm_source"] == "gemini").sum())
        if gemini_n > 0:
            st.success(f"Pipeline complete! {gemini_n} real Gemini explanations generated.")
        else:
            st.warning(
                "Daily Gemini quota exhausted (20 req/day on free tier). "
                "Quota resets at midnight Pacific time — run again tomorrow to get real AI explanations. "
                "All results are shown below using template explanations in the meantime."
            )
    except Exception as e:
        st.error(f"Explanation step failed: {e}")
        st.stop()

if "genai_df" not in st.session_state:
    st.info("Click **Run Pipeline** in the sidebar to start.")
    st.stop()

genai_df = st.session_state["genai_df"]


# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab_overview, tab_dq, tab_anomalies, tab_genai, tab_risk = st.tabs([
    "Overview", "Data Quality", "Anomalies", "AI Explanations", "Risk & Decisions"
])


# ── TAB 1: OVERVIEW ──────────────────────────
with tab_overview:
    st.subheader("Pipeline Summary")

    total_tx       = len(df)
    total_anomalies = len(unified)
    high_risk_count = (risk_table["risk_level"] == "HIGH").sum()
    detection_rate  = round(total_anomalies / total_tx * 100, 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions",  f"{total_tx:,}")
    c2.metric("Anomalies Detected",  f"{total_anomalies:,}")
    c3.metric("High-Risk Entities",  f"{int(high_risk_count)}")
    c4.metric("Detection Rate",      f"{detection_rate}%")

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        sev_counts = unified["severity"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]
        fig_sev = px.pie(
            sev_counts, names="Severity", values="Count",
            title="Anomaly Severity Distribution",
            color="Severity",
            color_discrete_map=SEVERITY_COLORS,
            hole=0.4,
        )
        fig_sev.update_layout(margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig_sev, use_container_width=True)

    with col_right:
        type_counts = unified["anomaly_type"].value_counts().reset_index()
        type_counts.columns = ["Anomaly Type", "Count"]
        TYPE_COLORS = {
            "amount_zscore":         "#FF4B4B",
            "account_amount_outlier":"#FF8C00",
            "device_churn":          "#4B9FFF",
            "ip_churn":              "#00CED1",
        }
        fig_type = px.bar(
            type_counts, x="Count", y="Anomaly Type",
            orientation="h", title="Anomaly Type Breakdown",
            color="Anomaly Type",
            color_discrete_map=TYPE_COLORS,
        )
        fig_type.update_layout(margin=dict(t=40, b=0, l=0, r=0), showlegend=False)
        st.plotly_chart(fig_type, use_container_width=True)

    st.divider()
    risk_counts = risk_table["risk_level"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
    fig_risk = go.Figure(go.Bar(
        x=risk_counts.index,
        y=risk_counts.values,
        marker_color=[SEVERITY_COLORS[r] for r in risk_counts.index],
        text=risk_counts.values,
        textposition="outside",
    ))
    fig_risk.update_layout(
        title="Entity Risk Level Distribution",
        xaxis_title="Risk Level", yaxis_title="Entity Count",
        margin=dict(t=50, b=0),
    )
    st.plotly_chart(fig_risk, use_container_width=True)


# ── TAB 2: DATA QUALITY ──────────────────────
with tab_dq:
    st.subheader("Data Quality Check Results")

    pass_count = (dq_results["status"] == "PASS").sum()
    fail_count = (dq_results["status"] == "FAIL").sum()
    flag_count = (dq_results["status"] == "FLAG").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Passed", f"{pass_count}", delta=None)
    c2.metric("Failed", f"{fail_count}", delta=f"-{fail_count}" if fail_count else None)
    c3.metric("Flagged", f"{flag_count}")

    st.divider()

    def _color_status(val):
        color = STATUS_COLORS.get(val, "white")
        return f"background-color: {color}22; color: {color}; font-weight: bold"

    def _color_severity(val):
        color = SEVERITY_COLORS.get(val, "white")
        return f"color: {color}; font-weight: bold"

    styled = dq_results.style \
        .applymap(_color_status, subset=["status"]) \
        .applymap(_color_severity, subset=["severity"])

    st.dataframe(styled, use_container_width=True, hide_index=True)

    if fail_count > 0:
        st.error(f"{fail_count} rule(s) FAILED — review the flagged rows above.")


# ── TAB 3: ANOMALIES ─────────────────────────
with tab_anomalies:
    st.subheader("Detected Anomalies")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        sev_filter = st.multiselect(
            "Filter by Severity", ["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM", "LOW"]
        )
    with col_f2:
        type_filter = st.multiselect(
            "Filter by Anomaly Type", unified["anomaly_type"].unique().tolist(),
            default=unified["anomaly_type"].unique().tolist()
        )

    filtered = unified[
        unified["severity"].isin(sev_filter) &
        unified["anomaly_type"].isin(type_filter)
    ]

    st.caption(f"Showing {len(filtered)} of {len(unified)} anomalies")

    def _highlight_severity_row(row):
        color = SEVERITY_COLORS.get(row["severity"], "#ffffff")
        return [f"background-color: {color}22"] * len(row)

    styled_anom = filtered.style.apply(_highlight_severity_row, axis=1)
    st.dataframe(styled_anom, use_container_width=True, hide_index=True)

    st.divider()
    st.download_button(
        "Download Anomalies CSV",
        data=unified.to_csv(index=False),
        file_name="unified_anomalies.csv",
        mime="text/csv",
    )


# ── TAB 4: AI EXPLANATIONS ───────────────────
with tab_genai:
    st.subheader("AI-Generated Explanations")

    # Show actual per-row source breakdown
    if "llm_source" in genai_df.columns:
        source_counts = genai_df["llm_source"].value_counts()
        gemini_count  = source_counts.get("gemini", 0)
        mock_count    = source_counts.get("mock", 0)
        if gemini_count > 0 and mock_count == 0:
            st.success(f"All {gemini_count} explanations generated by Gemini")
        elif gemini_count > 0:
            st.warning(f"{gemini_count} from Gemini, {mock_count} fell back to mock")
        else:
            st.warning("All explanations are mock — Gemini was not reached")

    search = st.text_input("Search by entity ID or anomaly type", "")

    display_df = genai_df.copy()
    if search.strip():
        mask = (
            display_df["entity_id"].astype(str).str.contains(search, case=False) |
            display_df["anomaly_type"].str.contains(search, case=False)
        )
        display_df = display_df[mask]

    st.caption(f"Showing {len(display_df)} explanations")

    for _, row in display_df.head(50).iterrows():
        sev_color = SEVERITY_COLORS.get(row["severity"], "#ccc")
        src = row.get("llm_source", "mock")
        src_badge = "🤖 Gemini" if src == "gemini" else ("🤖 OpenAI" if src == "openai" else "📝 Mock")
        with st.expander(
            f"**{row['anomaly_id']}** — {row['anomaly_type']} | "
            f"Entity: {row['entity_id']} | "
            f"Severity: {row['severity']} | {src_badge}"
        ):
            st.markdown(
                f"<span style='color:{sev_color}; font-weight:bold'>● {row['severity']}</span> &nbsp; "
                f"`{row['anomaly_type']}` &nbsp; Entity: `{row['entity_id']}`",
                unsafe_allow_html=True,
            )
            st.markdown("**AI Explanation:**")
            st.info(row["llm_explanation"])

    if len(display_df) > 50:
        st.caption(f"Showing first 50 of {len(display_df)}. Download full CSV below.")

    st.download_button(
        "Download Explanations CSV",
        data=genai_df.to_csv(index=False),
        file_name="genai_explanations.csv",
        mime="text/csv",
    )


# ── TAB 5: RISK & DECISIONS ──────────────────
with tab_risk:
    st.subheader("Risk Scoring & Agent Decisions")

    h = (risk_table["risk_level"] == "HIGH").sum()
    m = (risk_table["risk_level"] == "MEDIUM").sum()
    lo = (risk_table["risk_level"] == "LOW").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("High Risk Entities",   int(h))
    c2.metric("Medium Risk Entities", int(m))
    c3.metric("Low Risk Entities",    int(lo))

    st.divider()
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("**Risk Score Table**")
        def _highlight_risk(row):
            color = SEVERITY_COLORS.get(row["risk_level"], "#ffffff")
            return [f"background-color: {color}22"] * len(row)
        styled_risk = risk_table.style.apply(_highlight_risk, axis=1)
        st.dataframe(styled_risk, use_container_width=True, hide_index=True)

    with col_right:
        st.markdown("**Agent Decisions**")
        risk_filter = st.multiselect(
            "Filter by Risk Level", ["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM", "LOW"], key="risk_filter"
        )
        filtered_dec = decisions_df[decisions_df["risk_level"].isin(risk_filter)]

        def _highlight_decision(row):
            color = SEVERITY_COLORS.get(row["risk_level"], "#ffffff")
            return [f"background-color: {color}22"] * len(row)

        styled_dec = filtered_dec.style.apply(_highlight_decision, axis=1)
        st.dataframe(styled_dec, use_container_width=True, hide_index=True)

    st.divider()
    st.download_button(
        "Download Agent Decisions CSV",
        data=decisions_df.to_csv(index=False),
        file_name="agent_decisions.csv",
        mime="text/csv",
    )
