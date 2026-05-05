import os
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False


class GenAIExplanationEngine:
    """
    Standalone GenAI explanation engine using direct OpenAI/LangChain calls.
    Fallback-safe: uses mock responses when API key is absent.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 200,
    ):
        self.use_live_llm = False

        if LANGCHAIN_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                self.llm = ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self.prompt = PromptTemplate(
                    input_variables=["entity_type", "entity_id", "anomaly_type", "severity", "reason"],
                    template=(
                        "You are a financial risk analyst.\n"
                        "Entity: {entity_type} {entity_id}\n"
                        "Anomaly: {anomaly_type} | Severity: {severity}\n"
                        "Reason: {reason}\n\n"
                        "1. Explain why this anomaly is suspicious in simple terms.\n"
                        "2. Suggest what a risk analyst should investigate next.\n"
                        "Keep it concise and professional."
                    ),
                )
                self.use_live_llm = True
            except Exception:
                self.use_live_llm = False

    def _mock_response(self, anomaly: dict) -> str:
        return (
            f"This {anomaly['anomaly_type']} anomaly indicates unusual behavior. "
            f"The severity is {anomaly['severity']}, suggesting a meaningful deviation "
            f"from normal patterns. A risk analyst should review recent activity, "
            f"correlate with other anomalies, and determine whether further investigation "
            f"or customer verification is required."
        )

    def explain_anomaly(self, anomaly: dict) -> dict:
        if self.use_live_llm:
            try:
                prompt_text = self.prompt.format(**{k: anomaly[k] for k in ["entity_type", "entity_id", "anomaly_type", "severity", "reason"]})
                response = self.llm.invoke(prompt_text)
                explanation = response.content
            except Exception:
                explanation = self._mock_response(anomaly)
        else:
            explanation = self._mock_response(anomaly)

        return {
            "anomaly_id": anomaly["anomaly_id"],
            "entity_type": anomaly["entity_type"],
            "entity_id": anomaly["entity_id"],
            "anomaly_type": anomaly["anomaly_type"],
            "severity": anomaly["severity"],
            "llm_explanation": explanation,
        }

    def generate_explanations(self, unified_anomalies: pd.DataFrame) -> pd.DataFrame:
        results = [self.explain_anomaly(row) for _, row in unified_anomalies.iterrows()]
        return pd.DataFrame(results)
