import pandas as pd


class RiskAggregator:
    def aggregate(self, unified_anomalies: pd.DataFrame) -> pd.DataFrame:
        severity_weights = {
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
        }

        df = unified_anomalies.copy()
        df["severity_score"] = df["severity"].map(severity_weights)

        risk_table = (
            df.groupby("entity_id")
            .agg(
                anomaly_count=("anomaly_id", "count"),
                total_risk=("severity_score", "sum"),
                high_severity_count=("severity_score", lambda x: (x == 3).sum()),
            )
            .reset_index()
            .rename(columns={"entity_id": "EntityID"})
        )

        # Thresholds reflect max possible per entity type:
        # transaction entities top out at ~6 (two HIGH anomalies = 3+3)
        # account entities top out at ~4 (device_churn + ip_churn = 2+2)
        risk_table["risk_level"] = risk_table["total_risk"].apply(
            lambda x: "HIGH" if x >= 6 else "MEDIUM" if x >= 3 else "LOW"
        )

        return risk_table
