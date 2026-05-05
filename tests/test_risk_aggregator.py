import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agentic.risk_aggregator import RiskAggregator


def _make_anomalies(entity_severities: list) -> pd.DataFrame:
    rows = []
    for i, (entity_id, severity) in enumerate(entity_severities):
        rows.append({
            "anomaly_id": f"A{i:04d}",
            "entity_type": "account",
            "entity_id": entity_id,
            "anomaly_type": "test",
            "severity": severity,
            "reason": "test reason",
        })
    return pd.DataFrame(rows)


def test_high_risk_threshold():
    # Two HIGH anomalies = 3+3=6 >= 6 → HIGH (mirrors real max for transaction entities)
    df = _make_anomalies([("E1", "HIGH"), ("E1", "HIGH")])
    result = RiskAggregator().aggregate(df).set_index("EntityID")
    assert result.loc["E1", "risk_level"] == "HIGH"


def test_medium_risk_threshold():
    # One HIGH = 3 >= 3 → MEDIUM; two MEDIUM = 2+2=4 >= 3 → MEDIUM
    df = _make_anomalies([("E2", "MEDIUM"), ("E2", "MEDIUM")])
    result = RiskAggregator().aggregate(df).set_index("EntityID")
    assert result.loc["E2", "risk_level"] == "MEDIUM"


def test_low_risk_threshold():
    # One LOW = 1 < 3 → LOW
    df = _make_anomalies([("E3", "LOW")])
    result = RiskAggregator().aggregate(df).set_index("EntityID")
    assert result.loc["E3", "risk_level"] == "LOW"


def test_boundary_high():
    # total_risk = 6 exactly → HIGH
    df = _make_anomalies([("E4", "HIGH"), ("E4", "HIGH")])
    result = RiskAggregator().aggregate(df).set_index("EntityID")
    assert result.loc["E4", "risk_level"] == "HIGH"


def test_boundary_medium():
    # total_risk = 3 exactly (one HIGH) → MEDIUM
    df = _make_anomalies([("E5", "HIGH")])
    result = RiskAggregator().aggregate(df).set_index("EntityID")
    assert result.loc["E5", "risk_level"] == "MEDIUM"


def test_multiple_entities_independent():
    df = _make_anomalies([
        ("E1", "HIGH"), ("E1", "HIGH"),
        ("E2", "LOW"),
    ])
    result = RiskAggregator().aggregate(df).set_index("EntityID")
    assert result.loc["E1", "risk_level"] == "HIGH"
    assert result.loc["E2", "risk_level"] == "LOW"


def test_output_columns():
    df = _make_anomalies([("E1", "MEDIUM")])
    result = RiskAggregator().aggregate(df)
    assert set(["EntityID", "anomaly_count", "total_risk", "high_severity_count", "risk_level"]).issubset(result.columns)
