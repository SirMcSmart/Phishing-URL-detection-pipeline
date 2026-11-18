import math
from datetime import datetime, timezone

import dns.exception
import dns.resolver
import pytest

from phishdetect import dns_features, whois_features
from phishdetect.features import TTLCache, enrich, host_and_domain
from phishdetect.url_features import (DOMAIN_FEATURES, LEXICAL_FEATURES, InvalidURLError, extract_domain,
                                      extract_lexical, normalize_url, registered_domain, shannon_entropy)


# ----------------------------------------------------------------------------- URL parsing
@pytest.mark.parametrize("bad", ["", "   ", "javascript:alert(1)", "ftp://x.com/a", "file:///etc/passwd",
                                 "http://", "http://a.com/\r\nX-Injected: 1", "http://" + "a" * 3000 + ".com"])
def test_normalize_rejects(bad):
    with pytest.raises(InvalidURLError):
        normalize_url(bad)


def test_normalize_adds_scheme():
    assert normalize_url(" example.com/login ") == "http://example.com/login"


def test_lexical_vector_complete_and_finite():
    f = extract_lexical("http://paypal.com.secure-login.verify-account.xyz/webscr?cmd=_login&e=a@b.com")
    assert list(f) == list(LEXICAL_FEATURES)
    assert all(math.isfinite(v) for v in f.values())
    assert f["brand_outside_reg_domain"] == 1 and f["tld_is_risky"] == 1 and f["has_email_in_url"] == 1


def test_ip_hosts_detected():
    assert extract_lexical("http://192.168.10.5/x")["is_ip_host"] == 1
    assert extract_lexical("http://3232235777/x")["is_ip_host"] == 1  # decimal-encoded IPv4
    assert extract_domain("http://[2001:db8::1]/")["d_is_ip"] == 1


def test_brand_on_its_own_domain_is_not_flagged():
    assert extract_lexical("https://www.paypal.com/signin")["brand_outside_reg_domain"] == 0
    assert extract_domain("https://www.paypal.com")["d_brand_with_extra"] == 0
    assert extract_domain("https://paypal-account-check.com")["d_brand_with_extra"] == 1


def test_private_suffix_tenant_is_the_site():
    assert registered_domain("https://evil-login.web.app/x") == "evil-login.web.app"
    site, icann, is_ip = host_and_domain("https://evil-login.web.app/x")
    assert (site, icann, is_ip) == ("evil-login.web.app", "web.app", False)
    assert extract_domain("https://evil-login.web.app")["d_on_private_suffix"] == 1


def test_domain_features_ignore_dataset_artifacts():
    """Guard against the PhiUSIIL shortcut: scheme, www, path and query must not move
    the domain-level vector, otherwise the model can learn the collection format."""
    base = extract_domain("https://example-shop.com")
    for variant in ["http://example-shop.com", "https://www.example-shop.com", "http://www.example-shop.com/",
                    "https://example-shop.com/a/b/c.php?x=1&y=2#frag", "example-shop.com"]:
        assert extract_domain(variant) == base, variant
    assert list(base) == list(DOMAIN_FEATURES)


def test_entropy():
    assert shannon_entropy("") == 0
    assert shannon_entropy("aaaa") == 0
    assert shannon_entropy("ab") == pytest.approx(1.0)


# ----------------------------------------------------------------------------- WHOIS
def test_whois_date_features():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    f = whois_features._features_from_dates(datetime(2025, 12, 22, tzinfo=timezone.utc),
                                            datetime(2026, 12, 22, tzinfo=timezone.utc), None, now)
    assert f["whois_found"] == 1 and f["domain_age_days"] == pytest.approx(10)
    assert f["registration_period_days"] == pytest.approx(365)
    assert math.isnan(f["days_since_update"])


@pytest.mark.parametrize("raw", ["2020-05-01T10:00:00Z", "2020-05-01", "2020-05-01 10:00:00",
                                 ["2021-01-01", "2020-05-01"], datetime(2020, 5, 1)])
def test_parse_date_variants(raw):
    d = whois_features._parse_date(raw)
    assert d is not None and d.year == 2020 and d.tzinfo is not None


def test_whois_lookup_maps_failures_to_nan(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("down")
    monkeypatch.setattr(whois_features, "rdap_lookup", boom)
    feats, status = whois_features.lookup("example.com")
    assert status.startswith("error") and all(math.isnan(v) for v in feats.values())


# ----------------------------------------------------------------------------- DNS
class FakeResolver:
    def __init__(self, behaviour):
        self.behaviour = behaviour

    def resolve(self, name, rtype, raise_on_no_answer=False, tcp=False):
        result = self.behaviour(name, rtype, tcp)
        if isinstance(result, Exception):
            raise result
        return result


def test_dns_nxdomain_is_an_answer_not_an_error():
    feats, status = dns_features.lookup("gone.com", "gone.com", FakeResolver(lambda n, t, tcp: dns.resolver.NXDOMAIN()))
    assert status == "nxdomain"
    assert feats["dns_resolves"] == 0 and feats["n_mx_records"] == 0


def test_dns_timeout_gives_unknown():
    feats, status = dns_features.lookup("x.com", "x.com", FakeResolver(lambda n, t, tcp: dns.exception.Timeout()))
    assert status == "error:timeout" and all(math.isnan(v) for v in feats.values())


def test_dns_retries_over_tcp_after_udp_timeout():
    class Ans:
        rrset = None
    seen = []

    def behaviour(name, rtype, tcp):
        seen.append(tcp)
        return Ans() if tcp else dns.exception.Timeout()

    feats, status = dns_features.lookup("x.com", "x.com", FakeResolver(behaviour))
    assert status == "ok_no_address" and seen[:2] == [False, True]


# ----------------------------------------------------------------------------- enrichment glue
def test_enrich_skips_ip_and_caches_answers(monkeypatch):
    calls = []
    monkeypatch.setattr(whois_features, "lookup", lambda d, timeout=5: (calls.append(d) or
                        ({k: 1.0 for k in whois_features.WHOIS_FEATURES}, "ok:rdap")))
    monkeypatch.setattr(dns_features, "lookup", lambda h, z, resolver=None:
                        ({k: 1.0 for k in dns_features.DNS_FEATURES}, "ok"))
    _, st = enrich("http://10.0.0.1/a")
    assert st == {"whois": "skipped:ip", "dns": "skipped:ip"}
    cache = TTLCache()
    enrich("https://a.example.com/x", cache=cache)
    enrich("https://b.example.com/y", cache=cache)  # same site domain -> cache hit
    assert calls == ["example.com"]


def test_ttl_cache_expiry_and_bound():
    c = TTLCache(maxsize=2, ttl=60)
    c.set("a", 1); c.set("b", 2); c.set("c", 3)
    assert c.get("a") is None and c.get("c") == 3
    c.ttl = -1
    c.set("d", 4)
    assert c.get("d") is None


@pytest.mark.network
def test_live_lookup_google():
    feats, status = enrich("https://www.google.com")
    assert status["whois"].startswith("ok") and status["dns"] == "ok"
    assert feats["domain_age_days"] > 365 * 20
