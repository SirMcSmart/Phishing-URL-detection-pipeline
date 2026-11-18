"""WHOIS-style registration features, RDAP first with a port-43 WHOIS fallback.

RDAP (RFC 9082/9083) is the structured successor to WHOIS. The registry RDAP
server for each TLD comes from the IANA bootstrap file. TLDs with no RDAP
service fall back to classic WHOIS through the `python-whois` package.

Result semantics, which matter for the model:
  * whois_found = 1  -> a registration record was returned
  * whois_found = 0  -> the registry answered "no such domain" (RDAP 404 / WHOIS no match)
  * all features NaN -> the lookup failed (timeout, rate limit, unsupported TLD).
    Unknown is kept distinct from "not registered".
"""
from __future__ import annotations

import concurrent.futures as cf
import logging
import math
import threading
import time
from datetime import datetime, timezone

import requests

log = logging.getLogger(__name__)

IANA_RDAP_BOOTSTRAP = "https://data.iana.org/rdap/dns.json"
WHOIS_FEATURES: tuple[str, ...] = (
    "whois_found", "domain_age_days", "days_to_expiry",
    "registration_period_days", "days_since_update",
)
NAN = float("nan")

_bootstrap: dict[str, str] | None = None
_bootstrap_lock = threading.Lock()
_session = requests.Session()
_session.headers.update({"Accept": "application/rdap+json, application/json",
                         "User-Agent": "phishing-url-detection/1.0 (research)"})


def _empty() -> dict[str, float]:
    return {k: NAN for k in WHOIS_FEATURES}


def _load_bootstrap(timeout: float) -> dict[str, str]:
    global _bootstrap
    with _bootstrap_lock:
        if _bootstrap is None:
            resp = _session.get(IANA_RDAP_BOOTSTRAP, timeout=timeout)
            resp.raise_for_status()
            mapping: dict[str, str] = {}
            for tlds, urls in resp.json()["services"]:
                base = next((u for u in urls if u.startswith("https://")), urls[0])
                for tld in tlds:
                    mapping[tld.lower()] = base if base.endswith("/") else base + "/"
            _bootstrap = mapping
        return _bootstrap


def _parse_date(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, list):
        dates = [d for d in (_parse_date(v) for v in value) if d]
        return min(dates) if dates else None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        s = value.strip().replace("Z", "+00:00")
        for fmt in (None, "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d-%b-%Y", "%Y.%m.%d"):
            try:
                d = datetime.fromisoformat(s) if fmt is None else datetime.strptime(s[:19], fmt)
                return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def _features_from_dates(created, expires, updated, now: datetime) -> dict[str, float]:
    def days(a, b):
        return (a - b).total_seconds() / 86400.0 if a and b else NAN

    return {
        "whois_found": 1.0,
        "domain_age_days": days(now, created),
        "days_to_expiry": days(expires, now),
        "registration_period_days": days(expires, created),
        "days_since_update": days(now, updated),
    }


def rdap_lookup(domain: str, timeout: float = 5.0, now: datetime | None = None) -> dict[str, float] | None:
    """Return features, or None if this TLD has no RDAP service."""
    now = now or datetime.now(timezone.utc)
    tld = domain.rsplit(".", 1)[-1].lower()
    base = _load_bootstrap(timeout).get(tld)
    if base is None:
        return None
    for attempt in range(3):
        resp = _session.get(f"{base}domain/{domain}", timeout=timeout)
        if resp.status_code == 429:  # rate-limited: back off politely
            try:
                wait = float(resp.headers.get("Retry-After", 2 * (attempt + 1)))
            except ValueError:  # Retry-After may also be an HTTP date
                wait = 2 * (attempt + 1)
            time.sleep(min(wait, 10.0))  # never let a server park a worker for long
            continue
        break
    if resp.status_code == 404:
        out = _empty()
        out["whois_found"] = 0.0
        return out
    resp.raise_for_status()
    events = {e.get("eventAction", "").lower(): e.get("eventDate") for e in resp.json().get("events", [])}
    return _features_from_dates(
        _parse_date(events.get("registration")),
        _parse_date(events.get("expiration")),
        _parse_date(events.get("last changed")),
        now,
    )


def whois43_lookup(domain: str, timeout: float = 5.0, now: datetime | None = None) -> dict[str, float]:
    import whois  # python-whois; imported lazily because it is only a fallback

    now = now or datetime.now(timezone.utc)
    try:
        rec = whois.whois(domain, quiet=True)
    except Exception as exc:  # python-whois signals "no match" through exceptions
        msg = f"{type(exc).__name__} {exc}".lower()
        if "no match" in msg or "notfound" in msg or "not found" in msg:
            out = _empty()
            out["whois_found"] = 0.0
            return out
        raise
    created = _parse_date(rec.get("creation_date"))
    if not rec.get("domain_name") and created is None:
        out = _empty()
        out["whois_found"] = 0.0
        return out
    return _features_from_dates(created, _parse_date(rec.get("expiration_date")),
                                _parse_date(rec.get("updated_date")), now)


_pool = cf.ThreadPoolExecutor(max_workers=8, thread_name_prefix="whois")


def lookup(domain: str, timeout: float = 5.0) -> tuple[dict[str, float], str]:
    """Return (features, status) where status is 'ok', 'not_found', 'error:<reason>'."""
    if not domain or "." not in domain:
        return _empty(), "error:no_registered_domain"
    try:
        feats = rdap_lookup(domain, timeout=timeout)
        source = "rdap"
        if feats is None:
            # python-whois has no reliable timeout of its own; bound it here.
            feats = _pool.submit(whois43_lookup, domain, timeout).result(timeout=timeout * 2)
            source = "whois"
    except cf.TimeoutError:
        return _empty(), "error:timeout"
    except Exception as exc:  # network errors, malformed responses
        log.debug("whois lookup failed for %s: %s", domain, exc)
        return _empty(), f"error:{type(exc).__name__}"
    if feats["whois_found"] == 0.0:
        return feats, f"not_found:{source}"
    if all(math.isnan(feats[k]) for k in WHOIS_FEATURES if k != "whois_found"):
        return feats, f"ok_no_dates:{source}"
    return feats, f"ok:{source}"
