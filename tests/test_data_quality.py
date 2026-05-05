import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quality_checks.data_quality_engine import DataQualityEngine


def _base_df():
    return pd.DataFrame({
        "TransactionID": ["T1", "T2", "T3"],
        "AccountID": ["A1", "A1", "A2"],
        "TransactionAmount": [100.0, 200.0, 50.0],
        "AccountBalance": [500.0, 500.0, 300.0],
        "TransactionDate": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "PreviousTransactionDate": pd.to_datetime(["2023-12-31", "2024-01-01", "2024-01-02"]),
        "LoginAttempts": [1, 2, 1],
        "DeviceID": ["D1", "D2", "D1"],
        "IP Address": ["1.1.1.1", "2.2.2.2", "1.1.1.1"],
    })


def _run(df):
    return DataQualityEngine(df).run_all_checks().set_index("rule_name")


def test_clean_data_all_pass():
    results = _run(_base_df())
    for _, row in results.iterrows():
        assert row["status"] in ("PASS", "FLAG"), f"{row.name} unexpected status"
    assert results.loc["transaction_id_unique", "status"] == "PASS"
    assert results.loc["transaction_amount_positive", "status"] == "PASS"


def test_duplicate_transaction_id():
    df = _base_df()
    df.loc[1, "TransactionID"] = "T1"
    results = _run(df)
    assert results.loc["transaction_id_unique", "status"] == "FAIL"
    assert results.loc["transaction_id_unique", "affected_rows"] == 1


def test_negative_amount():
    df = _base_df()
    df.loc[0, "TransactionAmount"] = -10.0
    results = _run(df)
    assert results.loc["transaction_amount_positive", "status"] == "FAIL"


def test_amount_exceeds_balance():
    df = _base_df()
    df.loc[0, "TransactionAmount"] = 9999.0
    results = _run(df)
    assert results.loc["amount_exceeds_account_balance", "status"] == "FLAG"


def test_future_transaction():
    df = _base_df()
    df.loc[0, "TransactionDate"] = pd.Timestamp("2099-01-01")
    results = _run(df)
    assert results.loc["transaction_date_not_future", "status"] == "FAIL"


def test_excessive_login_attempts():
    df = _base_df()
    df.loc[0, "LoginAttempts"] = 10
    results = _run(df)
    assert results.loc["excessive_login_attempts", "status"] == "FLAG"
