"""DNS features from live resolution of the URL's host and registered domain.

Only DNS queries are made. The URL itself is never fetched, so scoring a URL
cannot be used to make this service send HTTP requests to arbitrary targets.

Result semantics:
  * NXDOMAIN / NoAnswer are real answers and produce zero counts.
  * If the resolver itself fails (timeout, SERVFAIL on every query) all
    features are NaN, so "unknown" stays distinct from "does not resolve".
"""
from __future__ import annotations

import logging

import dns.exception
import dns.resolver

log = logging.getLogger(__name__)

DNS_FEATURES: tuple[str, ...] = (
    "dns_resolves", "n_a_records", "n_aaaa_records", "has_cname", "a_ttl",
    "n_mx_records", "n_ns_records", "n_txt_records", "has_spf", "has_dmarc",
)
NAN = float("nan")


def make_resolver(timeout: float = 3.0, nameservers: list[str] | None = None) -> dns.resolver.Resolver:
    """Resolver pinned to public nameservers by default, so results do not depend on
    whatever local/ISP resolver the host happens to use. Pass [] for the system resolver."""
    nameservers = DEFAULT_NAMESERVERS if nameservers is None else nameservers
    r = dns.resolver.Resolver(configure=not nameservers)
    if nameservers:
        r.nameservers = list(nameservers)
    r.timeout = min(timeout, 2.0)
    r.lifetime = timeout
    return r


DEFAULT_NAMESERVERS = ["1.1.1.1", "8.8.8.8"]


class _ResolverDown(Exception):
    pass


_NX = object()  # sentinel: the name does not exist


def _query(resolver, name: str, rtype: str):
    """Return (rrset | None | _NX, ttl). Raise _ResolverDown on timeout."""
    try:
        try:
            ans = resolver.resolve(name, rtype, raise_on_no_answer=False)
        except (dns.exception.Timeout, dns.resolver.LifetimeTimeout):
            # Lossy networks drop UDP under load; one retry over TCP separates a
            # flaky path from a domain whose nameservers really do not answer.
            ans = resolver.resolve(name, rtype, raise_on_no_answer=False, tcp=True)
        return (ans.rrset, ans.rrset.ttl) if ans.rrset is not None else (None, NAN)
    except dns.resolver.NXDOMAIN:
        return _NX, NAN
    except (dns.resolver.NoAnswer, dns.resolver.NoNameservers):
        # NoNameservers covers SERVFAIL from a dead/lame delegation, which is an
        # answer about the domain, not a failure of our resolver.
        return None, NAN
    except (dns.exception.Timeout, dns.resolver.LifetimeTimeout) as exc:
        raise _ResolverDown("timeout") from exc
    except dns.exception.DNSException:
        return None, NAN


def _count(rrset) -> float:
    return 0.0 if rrset is None or rrset is _NX else float(len(rrset))


def _txt(rrset) -> list[str]:
    if rrset is None or rrset is _NX:
        return []
    return [b"".join(r.strings).decode("utf-8", "replace").lower() for r in rrset]


def lookup(host: str, registered_domain: str, resolver=None) -> tuple[dict[str, float], str]:
    resolver = resolver or make_resolver()
    out = {k: NAN for k in DNS_FEATURES}
    zone = registered_domain or host
    try:
        a, a_ttl = _query(resolver, host, "A")
    except _ResolverDown:
        return out, "error:timeout"

    def q(name, rtype):
        # A slow secondary query only blanks its own feature(s).
        try:
            return _query(resolver, name, rtype)[0]
        except _ResolverDown:
            return "timeout"

    zone_nx = a is _NX and host == zone
    aaaa = None if a is _NX else q(host, "AAAA")
    cname = None if a is _NX else q(host, "CNAME")
    mx, ns, txt, dmarc = (None,) * 4 if zone_nx else (q(zone, "MX"), q(zone, "NS"), q(zone, "TXT"),
                                                       q(f"_dmarc.{zone}", "TXT"))

    def val(rr, fn):
        return NAN if isinstance(rr, str) else fn(rr)

    n_a = _count(a)
    n_aaaa = val(aaaa, _count)
    out.update({
        "dns_resolves": float(n_a + (0.0 if n_aaaa != n_aaaa else n_aaaa) > 0),
        "n_a_records": n_a,
        "n_aaaa_records": n_aaaa,
        "has_cname": val(cname, lambda r: float(r is not None and r is not _NX)),
        "a_ttl": float(a_ttl),
        "n_mx_records": val(mx, _count),
        "n_ns_records": val(ns, _count),
        "n_txt_records": val(txt, lambda r: float(len(_txt(r)))),
        "has_spf": val(txt, lambda r: float(any(v.startswith("v=spf1") for v in _txt(r)))),
        "has_dmarc": val(dmarc, lambda r: float(any(v.startswith("v=dmarc1") for v in _txt(r)))),
    })
    if a is _NX:
        return out, "nxdomain"
    return out, "ok" if out["dns_resolves"] else "ok_no_address"
