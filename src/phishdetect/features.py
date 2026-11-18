"""Combine lexical, WHOIS and DNS features into one feature vector."""
from __future__ import annotations

import concurrent.futures as cf
import threading
import time
from collections import OrderedDict
from urllib.parse import urlsplit

from . import dns_features, whois_features
from .url_features import DOMAIN_FEATURES, _is_ip, domain_parts, extract_domain, normalize_url

ENRICHED_FEATURES: tuple[str, ...] = whois_features.WHOIS_FEATURES + dns_features.DNS_FEATURES
FULL_FEATURES: tuple[str, ...] = DOMAIN_FEATURES + ENRICHED_FEATURES
NAN = float("nan")


class TTLCache:
    """Small thread-safe LRU cache with per-entry expiry."""

    def __init__(self, maxsize: int = 10_000, ttl: float = 3600.0):
        self.maxsize, self.ttl = maxsize, ttl
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            value, expires = item
            if expires < time.monotonic():
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key, value):
        with self._lock:
            self._data[key] = (value, time.monotonic() + self.ttl)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)


def host_and_domain(url: str) -> tuple[str, str, bool]:
    """(site_domain, icann_domain, is_ip). DNS is queried on the site domain and
    WHOIS/RDAP on the ICANN-registrable domain."""
    host = (urlsplit(normalize_url(url)).hostname or "").lower().rstrip(".")
    if _is_ip(host):
        return host, "", True
    _, site, _, icann, _ = domain_parts(host)
    return site, icann, False


def enrich(url: str, *, timeout: float = 5.0, resolver=None, cache: TTLCache | None = None,
           pool: cf.Executor | None = None) -> tuple[dict[str, float], dict[str, str]]:
    """Run WHOIS and DNS lookups (in parallel) for a URL's domain.

    Returns (features, status) with status like {"whois": "ok:rdap", "dns": "ok"}.
    IP-literal hosts have no domain to look up; they get NaN with status "skipped:ip".
    """
    site, icann, is_ip = host_and_domain(url)
    if is_ip:
        return {k: NAN for k in ENRICHED_FEATURES}, {"whois": "skipped:ip", "dns": "skipped:ip"}

    key = (site, icann)
    if cache is not None and (hit := cache.get(key)) is not None:
        return hit

    if pool is None:
        w_feats, w_status = whois_features.lookup(icann, timeout=timeout)
        d_feats, d_status = dns_features.lookup(site, site, resolver=resolver)
    else:
        w_fut = pool.submit(whois_features.lookup, icann, timeout)
        d_fut = pool.submit(dns_features.lookup, site, site, resolver)
        try:
            w_feats, w_status = w_fut.result(timeout=timeout * 2.5)
        except cf.TimeoutError:
            w_feats, w_status = {k: NAN for k in whois_features.WHOIS_FEATURES}, "error:timeout"
        try:
            d_feats, d_status = d_fut.result(timeout=timeout * 2.5)
        except cf.TimeoutError:
            d_feats, d_status = {k: NAN for k in dns_features.DNS_FEATURES}, "error:timeout"

    result = ({**w_feats, **d_feats}, {"whois": w_status, "dns": d_status})
    # Cache only answers, not transient failures, so a timeout is retried next time.
    if cache is not None and not (w_status.startswith("error") or d_status.startswith("error")):
        cache.set(key, result)
    return result


_CANARY_NAMES = ("example.com", "iana.org")
_canary_state = {"ok": True, "checked": 0.0}
_canary_lock = threading.Lock()


def network_ok(resolver=None, max_age: float = 30.0) -> bool:
    """Can we resolve well-known names right now? Cached for ``max_age`` seconds.

    Used to tell "this domain is dead" apart from "our network is down": during an
    outage DNS failures can look like negative answers (NoNameservers -> no address).
    """
    with _canary_lock:
        if time.monotonic() - _canary_state["checked"] < max_age:
            return _canary_state["ok"]
    resolver = resolver or dns_features.make_resolver(3.0)
    ok = False
    for name in _CANARY_NAMES:
        try:
            if resolver.resolve(name, "A", raise_on_no_answer=False).rrset is not None:
                ok = True
                break
        except Exception:
            continue
    with _canary_lock:
        _canary_state.update(ok=ok, checked=time.monotonic())
    return ok


def looks_negative_or_failed(status: dict[str, str]) -> bool:
    """Outcomes that a network outage could also produce."""
    return (status.get("dns") in ("ok_no_address", "nxdomain") or status.get("dns", "").startswith("error")
            or status.get("whois", "").startswith(("error", "not_found")))


def enrichment_usable(status: dict[str, str]) -> bool:
    """True if at least one of the two lookups returned a real answer."""
    return any(not s.startswith(("error", "skipped")) for s in status.values())


def build_vector(url: str, enriched: dict[str, float] | None) -> dict[str, float]:
    feats = extract_domain(url)
    if enriched is not None:
        feats.update(enriched)
    return feats
