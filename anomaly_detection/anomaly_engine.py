import pandas as pd
import numpy as np


class AnomalyEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # ======================================================
    # 1. TRANSACTION-LEVEL AMOUNT Z-SCORE ANOMALIES
    # ======================================================
    def detect_amount_zscore_anomalies(self, z_threshold=3.0):
        mean_amt = self.df["TransactionAmount"].mean()
        std_amt = self.df["TransactionAmount"].std()

        self.df["z_score"] = (self.df["TransactionAmount"] - mean_amt) / std_amt

        anomalies = self.df[self.df["z_score"].abs() >= z_threshold].copy()

        anomalies["anomaly_type"] = "amount_zscore"
        anomalies["severity"] = anomalies["z_score"].abs().apply(
            lambda x: "HIGH" if x >= 5 else "MEDIUM" if x >= 3.5 else "LOW"
        )
        anomalies["reason"] = anomalies["z_score"].apply(
            lambda x: f"Transaction amount deviates {round(x, 2)} std from mean"
        )

        return anomalies[
            [
                "TransactionID",
                "AccountID",
                "TransactionAmount",
                "z_score",
                "anomaly_type",
                "severity",
                "reason",
            ]
        ]

    # ======================================================
    # 2. ACCOUNT-LEVEL AMOUNT ANOMALIES
    # ======================================================
    def detect_account_level_amount_anomalies(self, multiplier=3.0):
        stats = (
            self.df.groupby("AccountID")["TransactionAmount"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "account_mean", "std": "account_std"})
        )

        merged = self.df.merge(stats, on="AccountID", how="left")
        merged["threshold"] = merged["account_mean"] + multiplier * merged["account_std"]

        anomalies = merged[merged["TransactionAmount"] > merged["threshold"]].copy()

        if anomalies.empty:
            return pd.DataFrame(
                columns=[
                    "TransactionID",
                    "AccountID",
                    "TransactionAmount",
                    "account_mean",
                    "account_std",
                    "anomaly_type",
                    "severity",
                    "reason",
                ]
            )

        anomalies["anomaly_type"] = "account_amount_outlier"
        anomalies["severity"] = "HIGH"
        anomalies["reason"] = "Transaction unusually high for this account"

        return anomalies[
            [
                "TransactionID",
                "AccountID",
                "TransactionAmount",
                "account_mean",
                "account_std",
                "anomaly_type",
                "severity",
                "reason",
            ]
        ]

    # ======================================================
    # 3. DEVICE CHURN ANOMALIES
    # ======================================================
    def detect_device_churn_anomalies(self, device_threshold=5):
        device_counts = (
            self.df.groupby("AccountID")["DeviceID"]
            .nunique()
            .reset_index(name="device_count")
        )

        anomalies = device_counts[device_counts["device_count"] >= device_threshold].copy()

        anomalies["anomaly_type"] = "device_churn"
        anomalies["severity"] = "MEDIUM"
        anomalies["reason"] = anomalies["device_count"].apply(
            lambda x: f"Account used {x} unique devices"
        )

        return anomalies

    # ======================================================
    # 4. IP CHURN ANOMALIES (SCHEMA-SAFE ✔)
    # ======================================================
    def detect_ip_churn_anomalies(self, ip_threshold=5):
        # 🔍 Try to detect IP column safely
        possible_ip_cols = [
            "IP Address",
            "IPAddress",
            "IP",
            "IP_Address",
            "SourceIP",
            "ClientIP",
        ]

        ip_col = next((c for c in possible_ip_cols if c in self.df.columns), None)

        if ip_col is None:
            print("⚠️ IP column not found — skipping IP churn detection")
            return pd.DataFrame(
                columns=["AccountID", "ip_count", "anomaly_type", "severity", "reason"]
            )

        ip_counts = (
            self.df.groupby("AccountID")[ip_col]
            .nunique()
            .reset_index(name="ip_count")
        )

        anomalies = ip_counts[ip_counts["ip_count"] >= ip_threshold].copy()

        anomalies["anomaly_type"] = "ip_churn"
        anomalies["severity"] = "MEDIUM"
        anomalies["reason"] = anomalies["ip_count"].apply(
            lambda x: f"Account used {x} unique IP addresses"
        )

        return anomalies

    # ======================================================
    # 5. UNIFIED ANOMALY TABLE
    # ======================================================
    def build_unified_anomaly_table(
        self,
        amount_anomalies,
        account_amount_anomalies,
        device_anomalies,
        ip_anomalies,
    ):
        records = []
        anomaly_id = 1

        # Transaction-level anomalies
        for _, row in amount_anomalies.iterrows():
            records.append(
                {
                    "anomaly_id": f"A{anomaly_id:04d}",
                    "entity_type": "transaction",
                    "entity_id": row["TransactionID"],
                    "anomaly_type": row["anomaly_type"],
                    "severity": row["severity"],
                    "reason": row["reason"],
                }
            )
            anomaly_id += 1

        # Account-level amount anomalies
        for _, row in account_amount_anomalies.iterrows():
            records.append(
                {
                    "anomaly_id": f"A{anomaly_id:04d}",
                    "entity_type": "transaction",
                    "entity_id": row["TransactionID"],
                    "anomaly_type": row["anomaly_type"],
                    "severity": row["severity"],
                    "reason": row["reason"],
                }
            )
            anomaly_id += 1

        # Device churn anomalies
        for _, row in device_anomalies.iterrows():
            records.append(
                {
                    "anomaly_id": f"A{anomaly_id:04d}",
                    "entity_type": "account",
                    "entity_id": row["AccountID"],
                    "anomaly_type": row["anomaly_type"],
                    "severity": row["severity"],
                    "reason": row["reason"],
                }
            )
            anomaly_id += 1

        # IP churn anomalies
        for _, row in ip_anomalies.iterrows():
            records.append(
                {
                    "anomaly_id": f"A{anomaly_id:04d}",
                    "entity_type": "account",
                    "entity_id": row["AccountID"],
                    "anomaly_type": row["anomaly_type"],
                    "severity": row["severity"],
                    "reason": row["reason"],
                }
            )
            anomaly_id += 1

        return pd.DataFrame(records)
