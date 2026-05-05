import sys
import os
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from anomaly_detection.anomaly_engine import AnomalyEngine


def _base_df(n=50, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "TransactionID": [f"T{i}" for i in range(n)],
        "AccountID": [f"A{i % 5}" for i in range(n)],
        "TransactionAmount": rng.normal(loc=200, scale=30, size=n),
        "DeviceID": [f"D{i % 3}" for i in range(n)],
        "IP Address": [f"10.0.0.{i % 4}" for i in range(n)],
    })


def test_zscore_detects_outlier():
    df = _base_df()
    df.loc[0, "TransactionAmount"] = 99999.0
    engine = AnomalyEngine(df)
    result = engine.detect_amount_zscore_anomalies(z_threshold=3.0)
    assert len(result) >= 1
    assert "T0" in result["TransactionID"].values


def test_zscore_clean_data_no_anomalies():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "TransactionID": [f"T{i}" for i in range(100)],
        "AccountID": ["A1"] * 100,
        "TransactionAmount": rng.normal(loc=500, scale=5, size=100),
        "DeviceID": ["D1"] * 100,
        "IP Address": ["1.1.1.1"] * 100,
    })
    engine = AnomalyEngine(df)
    result = engine.detect_amount_zscore_anomalies(z_threshold=3.0)
    assert len(result) == 0


def test_device_churn_flagged():
    df = pd.DataFrame({
        "TransactionID": [f"T{i}" for i in range(10)],
        "AccountID": ["A1"] * 10,
        "TransactionAmount": [100.0] * 10,
        "DeviceID": [f"D{i}" for i in range(10)],
        "IP Address": ["1.1.1.1"] * 10,
    })
    engine = AnomalyEngine(df)
    result = engine.detect_device_churn_anomalies(device_threshold=5)
    assert len(result) == 1
    assert result.iloc[0]["AccountID"] == "A1"


def test_ip_churn_detected_with_space_column():
    df = pd.DataFrame({
        "TransactionID": [f"T{i}" for i in range(10)],
        "AccountID": ["A1"] * 10,
        "TransactionAmount": [100.0] * 10,
        "DeviceID": ["D1"] * 10,
        "IP Address": [f"10.0.0.{i}" for i in range(10)],
    })
    engine = AnomalyEngine(df)
    result = engine.detect_ip_churn_anomalies(ip_threshold=5)
    assert len(result) == 1
    assert result.iloc[0]["AccountID"] == "A1"


def test_unified_table_schema():
    df = _base_df()
    df.loc[0, "TransactionAmount"] = 99999.0
    engine = AnomalyEngine(df)
    amount = engine.detect_amount_zscore_anomalies()
    account = engine.detect_account_level_amount_anomalies()
    device = engine.detect_device_churn_anomalies()
    ip = engine.detect_ip_churn_anomalies()
    unified = engine.build_unified_anomaly_table(amount, account, device, ip)

    assert set(["anomaly_id", "entity_type", "entity_id", "anomaly_type", "severity", "reason"]).issubset(unified.columns)
    assert unified["anomaly_id"].str.startswith("A").all()
    assert unified["severity"].isin(["LOW", "MEDIUM", "HIGH"]).all()
