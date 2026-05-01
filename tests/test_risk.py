from app.risk.model import predict


def test_low_risk_inputs():
    nlp_out = {"sentiment": {"label": "POSITIVE", "score": 0.9},
               "text_risk_sub": 0.05, "n_risk_terms_critical": 0,
               "n_risk_terms_high": 0, "entities": [], "text_length": 40}
    cv_out = {"anomaly_score": 0.05, "edge_density": 0.1, "hotpix_ratio": 0.001}
    out = predict(nlp_out, cv_out)
    assert out["risk_label"] in {"LOW", "MEDIUM"}
    assert 0.0 <= out["risk_score"] <= 100.0


def test_high_risk_inputs():
    nlp_out = {"sentiment": {"label": "NEGATIVE", "score": 0.95},
               "text_risk_sub": 0.9, "n_risk_terms_critical": 3,
               "n_risk_terms_high": 2,
               "entities": [{"text": "reactor", "label": "PRODUCT"}],
               "text_length": 200}
    cv_out = {"anomaly_score": 0.9, "edge_density": 0.4, "hotpix_ratio": 0.05}
    out = predict(nlp_out, cv_out)
    assert out["risk_label"] == "HIGH"
    assert out["risk_score"] >= 66
