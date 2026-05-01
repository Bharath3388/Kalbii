from app.nlp.pipeline import analyze


def test_neutral_text_low_risk():
    out = analyze("The weather today is fine and the meeting went well.")
    assert out["text_risk_sub"] < 0.3
    assert "sentiment" in out
    assert isinstance(out["entities"], list)


def test_critical_text_high_risk():
    out = analyze("URGENT: gas leak and fire detected near the reactor — fatal hazard!")
    assert out["text_risk_sub"] > 0.5
    cats = {h["category"] for h in out["risk_terms"]}
    assert "critical" in cats


def test_high_severity_text():
    out = analyze("There is a hairline crack and corrosion on the weld joint.")
    cats = {h["category"] for h in out["risk_terms"]}
    assert "high" in cats
    assert out["n_risk_terms_high"] >= 1


def test_keywords_extracted():
    out = analyze("The pressure valve on the main pipeline showed irregular vibration.")
    assert len(out["keywords"]) >= 1
