"""Load trained models and score URLs; attach human-readable indicators."""
from __future__ import annotations

import concurrent.futures as cf
import math
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from .dns_features import make_resolver
from .features import (TTLCache, build_vector, enrich, enrichment_usable, host_and_domain,
                       looks_negative_or_failed, network_ok)
from .url_features import extract_lexical


@dataclass
class LoadedModel:
    name: str
    model: object
    features: list[str]
    threshold: float
    version: str
    trained_at: str
    metrics: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "LoadedModel":
        b = joblib.load(path)  # only load artifacts you produced: joblib unpickles code
        return cls(b["name"], b["model"], b["features"], float(b["threshold"]), b["version"],
                   b["trained_at"], b.get("metrics", {}))

    def predict(self, vec: dict[str, float]) -> float:
        row = pd.DataFrame([[vec.get(f, math.nan) for f in self.features]], columns=self.features)
        return float(self.model.predict_proba(row)[0, 1])


def indicators(url: str, vec: dict[str, float]) -> list[str]:
    """Rule-based observations shown to an analyst. These are heuristics, not an
    explanation of the model's score, and some (path/query) are not model inputs."""
    lex = extract_lexical(url)
    out = []
    age = vec.get("domain_age_days", math.nan)
    if age == age and age < 30:
        out.append(f"domain registered {int(age)} days ago")
    elif age == age and age < 365:
        out.append(f"domain is less than a year old ({int(age)} days)")
    if vec.get("whois_found") == 0:
        out.append("registry has no record of this domain")
    if vec.get("dns_resolves") == 0:
        out.append("domain does not resolve in DNS")
    if vec.get("dns_resolves") == 1 and vec.get("n_mx_records") == 0:
        out.append("domain has no mail (MX) records")
    if lex["is_ip_host"]:
        out.append("host is a raw IP address")
    if lex["brand_outside_reg_domain"]:
        out.append("brand name appears outside the registered domain")
    if vec.get("d_brand_with_extra"):
        out.append("domain combines a brand name with extra words")
    if vec.get("d_on_private_suffix"):
        out.append("hosted on a shared/free hosting suffix")
    if lex["tld_is_risky"]:
        out.append("TLD is frequently abused")
    if lex["has_punycode"]:
        out.append("punycode (possible homograph) in host")
    if lex["n_at"]:
        out.append("'@' in URL can hide the real destination")
    if lex["is_shortener"]:
        out.append("URL shortener hides the destination")
    if lex["n_suspicious_keywords"] >= 2:
        out.append("several credential/urgency keywords in URL")
    if lex["has_email_in_url"]:
        out.append("email address embedded in URL")
    if lex["n_subdomain_levels"] >= 3:
        out.append(f"deep subdomain nesting ({int(lex['n_subdomain_levels'])} levels)")
    return out


class Scorer:
    def __init__(self, models_dir: Path, *, timeout: float = 4.0, nameservers=None,
                 cache_ttl: float = 3600.0, cache_size: int = 10_000):
        self.lexical = LoadedModel.load(models_dir / "lexical.joblib")
        full_path = models_dir / "full.joblib"
        self.full = LoadedModel.load(full_path) if full_path.exists() else None
        self.timeout = timeout
        self.resolver = make_resolver(timeout, nameservers)
        self.cache = TTLCache(cache_size, cache_ttl)
        self.pool = cf.ThreadPoolExecutor(max_workers=16, thread_name_prefix="enrich")

    def score(self, url: str, enrich_lookups: bool = True) -> dict:
        site, icann, _ = host_and_domain(url)
        status = {"whois": "skipped:disabled", "dns": "skipped:disabled"}
        enriched = None
        if enrich_lookups and self.full is not None:
            enriched, status = enrich(url, timeout=self.timeout, resolver=self.resolver,
                                      cache=self.cache, pool=self.pool)
            if looks_negative_or_failed(status) and not network_ok(self.resolver):
                # Our own network is down: "does not resolve" would be a false signal.
                status = {"whois": "error:network_down", "dns": "error:network_down"}
        use_full = enriched is not None and enrichment_usable(status)
        model = self.full if use_full else self.lexical
        vec = build_vector(url, enriched if use_full else None)
        p = model.predict(vec)
        verdict = "phishing" if p >= model.threshold else "legitimate"
        return {
            "url": url,
            "site_domain": site,
            "registered_domain": icann or site,
            "score": round(p, 6),
            "threshold": round(model.threshold, 6),
            "verdict": verdict,
            "model": model.name,
            "model_version": model.version,
            "lookups": status,
            "indicators": indicators(url, vec),
            "enrichment": ({k: (None if v != v else v) for k, v in enriched.items()} if use_full else None),
        }

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
