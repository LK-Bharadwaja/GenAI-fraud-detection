import pandas as pd


class DataQualityEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = []

    # ------------------------
    # Internal helpers
    # ------------------------
    def _add_result(self, rule_name, category, affected_rows, severity, status):
        self.results.append({
            "rule_name": rule_name,
            "category": category,
            "affected_rows": int(affected_rows),
            "severity": severity,
            "status": status
        })

    def _ensure_datetime(self, column_name):
        self.df[column_name] = pd.to_datetime(
            self.df[column_name],
            errors="coerce"
        )

    # ------------------------
    # INTEGRITY RULES
    # ------------------------
    def rule_transaction_id_unique(self):
        duplicates = self.df["TransactionID"].duplicated().sum()
        self._add_result(
            "transaction_id_unique",
            "integrity",
            duplicates,
            "HIGH",
            "FAIL" if duplicates > 0 else "PASS"
        )

    # ------------------------
    # FINANCIAL RULES
    # ------------------------
    def rule_positive_transaction_amount(self):
        invalid = (self.df["TransactionAmount"] <= 0).sum()
        self._add_result(
            "transaction_amount_positive",
            "financial",
            invalid,
            "HIGH",
            "FAIL" if invalid > 0 else "PASS"
        )

    def rule_amount_exceeds_balance(self):
        violations = (self.df["TransactionAmount"] > self.df["AccountBalance"]).sum()
        self._add_result(
            "amount_exceeds_account_balance",
            "financial",
            violations,
            "MEDIUM",
            "FLAG" if violations > 0 else "PASS"
        )

    # ------------------------
    # TEMPORAL RULES
    # ------------------------
    def rule_transaction_not_future(self):
        self._ensure_datetime("TransactionDate")
        future = (self.df["TransactionDate"] > pd.Timestamp.now()).sum()
        self._add_result(
            "transaction_date_not_future",
            "temporal",
            future,
            "HIGH",
            "FAIL" if future > 0 else "PASS"
        )

    def rule_transaction_after_previous(self):
        self._ensure_datetime("TransactionDate")
        self._ensure_datetime("PreviousTransactionDate")

        violations = (
            self.df["TransactionDate"] < self.df["PreviousTransactionDate"]
        ).sum()

        self._add_result(
            "transaction_after_previous",
            "temporal",
            violations,
            "HIGH",
            "FAIL" if violations > 0 else "PASS"
        )

    # ------------------------
    # BEHAVIORAL RULES
    # ------------------------
    def rule_excessive_login_attempts(self, threshold=5):
        high_attempts = (self.df["LoginAttempts"] > threshold).sum()
        self._add_result(
            "excessive_login_attempts",
            "behavioral",
            high_attempts,
            "LOW",
            "FLAG" if high_attempts > 0 else "PASS"
        )

    def rule_multiple_devices(self, threshold=5):
        device_counts = self.df.groupby("AccountID")["DeviceID"].nunique()
        flagged = (device_counts > threshold).sum()
        self._add_result(
            "multiple_devices_per_account",
            "behavioral",
            flagged,
            "LOW",
            "FLAG" if flagged > 0 else "PASS"
        )

    def rule_multiple_ips(self, threshold=5):
        ip_counts = self.df.groupby("AccountID")["IP Address"].nunique()
        flagged = (ip_counts > threshold).sum()
        self._add_result(
            "multiple_ips_per_account",
            "behavioral",
            flagged,
            "LOW",
            "FLAG" if flagged > 0 else "PASS"
        )

    # ------------------------
    # RUN ALL RULES
    # ------------------------
    def run_all_checks(self):
        self.rule_transaction_id_unique()
        self.rule_positive_transaction_amount()
        self.rule_amount_exceeds_balance()
        self.rule_transaction_not_future()
        self.rule_transaction_after_previous()
        self.rule_excessive_login_attempts()
        self.rule_multiple_devices()
        self.rule_multiple_ips()

        return pd.DataFrame(self.results)
