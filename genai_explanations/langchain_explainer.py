import os
import time
import logging
import pandas as pd

log = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    load_dotenv(os.path.normpath(_env_path))
except ImportError:
    pass

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_core.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["anomaly_type", "severity", "reason"],
    template=(
        "You are a senior financial fraud analyst reviewing a bank transaction alert.\n\n"
        "Anomaly detected:\n"
        "- Type: {anomaly_type}\n"
        "- Severity: {severity}\n"
        "- Details: {reason}\n\n"
        "Write a 2-3 sentence explanation of why this is suspicious and what a human analyst "
        "should do next. Be specific, conversational, and avoid generic phrases. "
        "Do not repeat the input fields verbatim."
    ),
) if LANGCHAIN_AVAILABLE else None

# Only gemini-2.5-flash has free-tier quota on this project (others return limit:0)
GEMINI_MODELS = ["gemini-2.5-flash"]


class LangChainExplanationEngine:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.4,
        max_tokens: int = 300,
    ):
        self.llm = None
        self.llm_source = "mock"
        self.model_used = None

        # 1. Try Gemini (free) — try lite model first for higher rate limits
        if GEMINI_AVAILABLE and LANGCHAIN_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
            for model in GEMINI_MODELS:
                try:
                    self.llm = ChatGoogleGenerativeAI(
                        model=model,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        google_api_key=os.getenv("GOOGLE_API_KEY"),
                    )
                    self.llm_source = "gemini"
                    self.model_used = model
                    log.info(f"LLM initialised: Gemini ({model})")
                    break
                except Exception as e:
                    log.warning(f"Gemini init failed for {model}: {e}")
                    self.llm = None

        # 2. Fall back to OpenAI
        if self.llm is None and OPENAI_AVAILABLE and LANGCHAIN_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                self.llm = ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self.llm_source = "openai"
                self.model_used = model_name
                log.info("LLM initialised: OpenAI")
            except Exception as e:
                log.warning(f"OpenAI init failed: {e}")
                self.llm = None

        if self.llm is None:
            log.warning("No LLM available — using mock explanations")

    def _mock_explanation(self, row) -> str:
        return (
            f"This {row['anomaly_type']} anomaly was detected with severity "
            f"{row['severity']}. The system identified the issue due to: "
            f"{row['reason']}. Analysts should validate this activity against "
            f"historical behavior and apply risk mitigation controls if required."
        )

    def generate_explanations(
        self,
        unified_anomalies: pd.DataFrame,
        severity_filter: list = None,
        delay_seconds: float = 13.0,
        max_llm_calls: int = 30,
        progress_callback=None,
    ) -> pd.DataFrame:
        """
        severity_filter: only call LLM for these severities — LOW rows get mock.
        delay_seconds:   min gap between LLM calls (gemini-2.5-flash = 5 req/min → 13s safe).
        max_llm_calls:   cap total LLM calls — rest get mock. Avoids long waits on free tier.
        progress_callback: called as callback(current, total) after each LLM row.
        """
        if severity_filter is None:
            severity_filter = ["HIGH", "MEDIUM"]

        llm_candidates = unified_anomalies[unified_anomalies["severity"].isin(severity_filter)]
        # Prioritise HIGH over MEDIUM so the cap picks the most important ones
        llm_candidates = llm_candidates.sort_values(
            "severity", key=lambda s: s.map({"HIGH": 0, "MEDIUM": 1}), kind="stable"
        )
        llm_ids = set(llm_candidates.head(max_llm_calls)["anomaly_id"])
        total_llm = len(llm_ids)
        log.info(
            f"Generating explanations via {self.llm_source} | "
            f"{total_llm} LLM calls (capped at {max_llm_calls}) + "
            f"{len(unified_anomalies) - total_llm} mock"
        )

        results_map = {}
        llm_done = 0
        llm_errors = 0
        daily_quota_hit = False  # stop all LLM calls once daily limit is detected

        for _, row in unified_anomalies.iterrows():
            row_id = row["anomaly_id"]
            use_llm = (
                self.llm is not None
                and PROMPT_TEMPLATE is not None
                and row_id in llm_ids
                and not daily_quota_hit
            )

            if use_llm:
                t_start = time.time()
                try:
                    prompt_text = PROMPT_TEMPLATE.format(
                        anomaly_type=row["anomaly_type"],
                        severity=row["severity"],
                        reason=row["reason"],
                    )
                    response = self.llm.invoke(prompt_text)
                    explanation = response.content.strip()
                    source = self.llm_source
                except Exception as e:
                    error_str = str(e)
                    llm_errors += 1
                    # Detect daily quota exhaustion — no point retrying further rows
                    if "GenerateRequestsPerDayPerProjectPerModel" in error_str:
                        daily_quota_hit = True
                        log.warning("Daily Gemini quota exhausted — switching remaining rows to mock")
                    else:
                        log.warning(f"LLM call failed for {row_id}: {e}")
                    explanation = self._mock_explanation(row)
                    source = "mock"

                # Rate limiting: sleep only the remaining time to fill the gap
                if not daily_quota_hit:
                    elapsed = time.time() - t_start
                    wait = delay_seconds - elapsed
                    if wait > 0:
                        time.sleep(wait)

                llm_done += 1
                if progress_callback:
                    progress_callback(llm_done, total_llm)
            else:
                explanation = self._mock_explanation(row)
                source = "mock"

            results_map[row_id] = {
                "anomaly_id":      row_id,
                "entity_type":     row["entity_type"],
                "entity_id":       row["entity_id"],
                "anomaly_type":    row["anomaly_type"],
                "severity":        row["severity"],
                "llm_explanation": explanation,
                "llm_source":      source,
            }

        if llm_errors:
            log.warning(f"{llm_errors}/{total_llm} LLM calls fell back to mock")

        rows = [results_map[aid] for aid in unified_anomalies["anomaly_id"]]
        return pd.DataFrame(rows)
