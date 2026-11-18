import math

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from phishdetect import api as api_module
from phishdetect import scorer as scorer_module
from phishdetect.features import ENRICHED_FEATURES, FULL_FEATURES
from phishdetect.train import evaluate, grouped_split, make_model, pick_threshold, save_model
from phishdetect.url_features import DOMAIN_FEATURES

MCFG = {"learning_rate": 0.1, "max_iter": 30, "max_leaf_nodes": 7, "min_samples_leaf": 5,
        "l2_regularization": 0.0, "early_stopping": False, "validation_fraction": None,
        "n_iter_no_change": 5, "threshold_strategy": "f1"}


# ----------------------------------------------------------------------------- training helpers
def test_grouped_split_has_no_domain_overlap():
    df = pd.DataFrame({"reg_domain": [f"d{i % 50}.com" for i in range(500)]})
    tr, va, te = grouped_split(df, 0.2, 0.2, seed=0)
    doms = [set(df.reg_domain.iloc[i]) for i in (tr, va, te)]
    assert not (doms[0] & doms[1] or doms[0] & doms[2] or doms[1] & doms[2])
    assert len(tr) + len(va) + len(te) == 500


def test_threshold_strategies():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9])
    t = pick_threshold(y, p, "f1")
    assert evaluate(y, p, t)["f1"] >= 0.85
    t0 = pick_threshold(y, p, 0.0)  # no false positives allowed
    assert evaluate(y, p, t0)["confusion"]["fp"] == 0


# ----------------------------------------------------------------------------- API
def _fit(features, seed):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(300, len(features))), columns=list(features))
    y = (X.iloc[:, 0] > 0).astype(int).to_numpy()
    return make_model(MCFG, seed).fit(X, y)


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    save_model(d / "lexical.joblib", _fit(DOMAIN_FEATURES, 1), DOMAIN_FEATURES, 0.5, "lexical", {"metrics": {}})
    save_model(d / "full.joblib", _fit(FULL_FEATURES, 2), FULL_FEATURES, 0.5, "full", {"metrics": {}})
    return d


@pytest.fixture()
def client(models_dir, monkeypatch):
    state = {"status": {"whois": "ok:rdap", "dns": "ok"}}

    def fake_enrich(url, **kw):
        feats = {k: 1.0 for k in ENRICHED_FEATURES}
        feats["domain_age_days"] = 3.0
        return feats, dict(state["status"])

    monkeypatch.setattr(scorer_module, "enrich", fake_enrich)
    monkeypatch.delenv("PHISHDETECT_API_KEY", raising=False)
    api_module.app.state.scorer = scorer_module.Scorer(models_dir)
    with TestClient(api_module.app) as c:
        c.state = state
        yield c
    api_module.app.state.scorer = None


def test_health_and_headers(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["models"] == {"lexical": True, "full": True}
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]


def test_score_uses_full_model_and_reports_indicators(client):
    r = client.post("/v1/score", json={"url": "http://paypal-verify-account.xyz/login"})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "full" and body["verdict"] in ("phishing", "legitimate")
    assert 0.0 <= body["score"] <= 1.0
    assert body["site_domain"] == "paypal-verify-account.xyz"
    assert any("registered 3 days ago" in i for i in body["indicators"])


def test_falls_back_to_lexical_when_lookups_fail(client):
    client.state["status"] = {"whois": "error:timeout", "dns": "error:timeout"}
    body = client.post("/v1/score", json={"url": "example.com"}).json()
    assert body["model"] == "lexical" and body["enrichment"] is None


def test_enrich_false_uses_lexical(client):
    body = client.post("/v1/score", json={"url": "https://example.com", "enrich": False}).json()
    assert body["model"] == "lexical" and body["lookups"]["whois"] == "skipped:disabled"


@pytest.mark.parametrize("payload", [{"url": "javascript:alert(1)"}, {"url": ""}, {"url": "x" * 5000}, {}])
def test_invalid_input_rejected(client, payload):
    assert client.post("/v1/score", json=payload).status_code == 422


def test_batch(client):
    r = client.post("/v1/score/batch", json={"urls": ["a.com", "ftp://bad", "https://b.org/x"], "enrich": False})
    res = r.json()["results"]
    assert r.status_code == 200 and len(res) == 3
    assert "error" in res[1] and res[0]["model"] == "lexical"
    assert client.post("/v1/score/batch", json={"urls": ["a.com"] * 51}).status_code == 422


def test_api_key_enforced(client, monkeypatch):
    monkeypatch.setenv("PHISHDETECT_API_KEY", "s3cret-test-key")
    assert client.post("/v1/score", json={"url": "a.com"}).status_code == 401
    assert client.post("/v1/score", json={"url": "a.com"}, headers={"X-API-Key": "wrong"}).status_code == 401
    ok = client.post("/v1/score", json={"url": "a.com"}, headers={"X-API-Key": "s3cret-test-key"})
    assert ok.status_code == 200
    assert client.get("/health").status_code == 200  # health stays open for probes


def test_rate_limiter():
    rl = api_module.RateLimiter(2)
    assert rl.allow("1.2.3.4") and rl.allow("1.2.3.4") and not rl.allow("1.2.3.4")
    assert rl.allow("5.6.7.8")
    assert api_module.RateLimiter(0).allow("x")


def test_model_info(client):
    info = client.get("/v1/model").json()
    assert set(info) == {"lexical", "full"} and not math.isnan(info["full"]["threshold"])


def test_network_outage_falls_back_to_lexical(client, monkeypatch):
    client.state["status"] = {"whois": "error:ConnectionError", "dns": "ok_no_address"}
    monkeypatch.setattr(scorer_module, "network_ok", lambda *a, **k: False)
    body = client.post("/v1/score", json={"url": "example.com"}).json()
    assert body["model"] == "lexical" and body["lookups"]["dns"] == "error:network_down"


def test_negative_answers_trusted_when_network_up(client, monkeypatch):
    client.state["status"] = {"whois": "not_found:rdap", "dns": "nxdomain"}
    monkeypatch.setattr(scorer_module, "network_ok", lambda *a, **k: True)
    assert client.post("/v1/score", json={"url": "example.com"}).json()["model"] == "full"
