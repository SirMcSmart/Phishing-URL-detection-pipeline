"""Lexical (string-only) URL features.

Everything here is computed from the URL string alone: no network access, no
page fetch. The same function is used at training time and at serving time, so
there is no training/serving skew for this feature group.
"""
from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from functools import lru_cache
from urllib.parse import urlsplit

import tldextract

# Bundled public-suffix snapshot only: deterministic and works offline / in Docker.
# Private suffixes (web.app, github.io, blogspot.com, ...) are enabled so that the
# "site" of https://evil.web.app is evil.web.app, the unit a phisher controls.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)

MAX_URL_LENGTH = 2048

SUSPICIOUS_KEYWORDS = (
    "login", "log-in", "signin", "sign-in", "verify", "verification", "account",
    "update", "secure", "security", "banking", "confirm", "password", "webscr",
    "wallet", "unlock", "suspend", "billing", "invoice", "support", "auth",
    "recover", "validate", "payment", "authenticate", "session", "token",
)

# Brands that phishing kits commonly impersonate. Used only to detect a brand name
# that appears somewhere in the URL *other than* the registered domain.
BRANDS = (
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook", "live",
    "google", "gmail", "amazon", "netflix", "facebook", "instagram", "whatsapp",
    "linkedin", "dropbox", "docusign", "adobe", "chase", "wellsfargo",
    "bankofamerica", "citi", "hsbc", "barclays", "santander", "usps", "dhl",
    "fedex", "ups", "irs", "coinbase", "binance", "metamask", "steam", "ebay",
    "walmart", "att", "verizon", "yahoo", "aol", "wetransfer", "sharepoint",
    "onedrive", "telegram",
)

SHORTENERS = frozenset({
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bitly.com", "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc",
    "rebrand.ly", "t.ly", "s.id", "v.gd", "qrco.de", "lnkd.in", "trib.al",
})

# TLDs frequently over-represented in abuse reports. Kept as a fixed list so the
# feature does not depend on the training data's TLD distribution.
RISKY_TLDS = frozenset({
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "buzz", "icu", "cyou", "rest",
    "monster", "click", "link", "work", "support", "live", "online", "site",
    "shop", "store", "fun", "space", "website", "zip", "mov", "cam", "quest",
    "sbs", "cfd", "bond", "lol", "pw", "cc", "ru", "cn", "su", "info", "biz",
})
COMMON_TLDS = frozenset({"com", "org", "net", "edu", "gov"})

_RISKY_EXTENSIONS = (".exe", ".zip", ".rar", ".apk", ".scr", ".js", ".bin", ".msi", ".iso")
_WEB_EXTENSIONS = (".php", ".html", ".htm", ".asp", ".aspx", ".jsp", ".cgi")
_VOWELS = set("aeiou")
_SPECIALS = set("-_.~!*'();:@&=+$,/?#[]%")

LEXICAL_FEATURES: tuple[str, ...] = (
    "url_len", "host_len", "path_len", "query_len", "fragment_len",
    "n_dots_url", "n_hyphens_url", "n_underscores_url", "n_slashes_url",
    "n_qmarks", "n_equals", "n_at", "n_ampersands", "n_tildes", "n_percent",
    "n_plus", "digit_ratio_url", "letter_ratio_url", "special_ratio_url",
    "url_entropy", "n_dots_host", "n_hyphens_host", "n_digits_host",
    "host_entropy", "n_subdomain_levels", "tld_len", "is_ip_host", "has_port",
    "is_https", "has_punycode", "path_depth", "has_double_slash_in_path",
    "has_web_extension", "has_risky_extension", "n_query_params",
    "n_suspicious_keywords", "brand_outside_reg_domain", "is_shortener",
    "tld_is_risky", "tld_is_common", "longest_token_len", "mean_token_len",
    "max_digit_run", "max_consonant_run_host", "reg_domain_len",
    "reg_domain_n_digits", "reg_domain_n_hyphens", "reg_domain_vowel_ratio",
    "has_hex_or_encoded", "has_email_in_url",
)


class InvalidURLError(ValueError):
    """The input cannot be parsed as an http(s) URL with a hostname."""


def normalize_url(raw: str) -> str:
    """Trim, add a scheme if missing, and reject anything that is not http(s)."""
    if not isinstance(raw, str):
        raise InvalidURLError("URL must be a string")
    url = raw.strip()
    if not url:
        raise InvalidURLError("URL is empty")
    if len(url) > MAX_URL_LENGTH:
        raise InvalidURLError(f"URL longer than {MAX_URL_LENGTH} characters")
    if any(c in url for c in "\r\n\t\x00"):
        raise InvalidURLError("URL contains control characters")
    m = re.match(r"([a-zA-Z][a-zA-Z0-9+.-]*):(.?)", url)
    # "javascript:..." / "data:..." / "ftp://..." carry a scheme; "example.com:8080/x" is host:port.
    if m and not m.group(2).isdigit():
        scheme = m.group(1).lower()
        if scheme not in ("http", "https") or not url[m.end(1):].startswith("://"):
            raise InvalidURLError(f"unsupported scheme: {scheme}")
    else:
        url = "http://" + url
    try:
        host = urlsplit(url).hostname
    except ValueError as exc:  # e.g. bad IPv6 literal or port
        raise InvalidURLError(str(exc)) from exc
    if not host:
        raise InvalidURLError("URL has no hostname")
    return url


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        pass
    # Decimal / hex encoded IPv4 such as http://3232235777/ or 0xC0A80001
    return bool(re.fullmatch(r"(0x[0-9a-f]+|\d{8,10})", host))


def _max_run(s: str, pred) -> int:
    best = cur = 0
    for ch in s:
        cur = cur + 1 if pred(ch) else 0
        best = max(best, cur)
    return best


@lru_cache(maxsize=65536)
def domain_parts(host: str) -> tuple[str, str, str, str, bool]:
    """(subdomain, site_domain, site_suffix, icann_domain, on_private_suffix).

    site_domain  : eTLD+1 including PSL private suffixes (evil.web.app)
    icann_domain : the domain a registrar actually sold (web.app); WHOIS/RDAP key
    """
    ext = _EXTRACT(host)
    site = ext.top_domain_under_public_suffix or host
    icann = ext.top_domain_under_registry_suffix or site
    return ext.subdomain, site, ext.suffix, icann, bool(ext.is_private)


def split_host(host: str) -> tuple[str, str, str]:
    """Return (subdomain, site_domain, suffix)."""
    sub, site, suffix, _, _ = domain_parts(host)
    return sub, site, suffix


def registered_domain(url: str) -> str:
    """Site domain (eTLD+1 incl. private suffixes) of a URL, or the bare host for IPs."""
    host = (urlsplit(normalize_url(url)).hostname or "").lower().rstrip(".")
    if _is_ip(host):
        return host
    return split_host(host)[1]


def extract_lexical(raw_url: str) -> dict[str, float]:
    url = normalize_url(raw_url)
    parts = urlsplit(url)
    host = (parts.hostname or "").lower().rstrip(".")
    path, query, fragment = parts.path or "", parts.query or "", parts.fragment or ""
    lower = url.lower()

    ip_host = _is_ip(host)
    if ip_host:
        subdomain, reg_domain, suffix = "", host, ""
    else:
        subdomain, reg_domain, suffix = split_host(host)
    reg_label = reg_domain[: -len(suffix) - 1] if suffix and reg_domain.endswith("." + suffix) else reg_domain
    tld = suffix.rsplit(".", 1)[-1] if suffix else ""

    n = len(url)
    digits = sum(ch.isdigit() for ch in url)
    letters = sum(ch.isalpha() for ch in url)
    specials = sum(ch in _SPECIALS for ch in url)
    tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", url) if t]

    outside = lower.replace(reg_domain, " ", 1) if reg_domain else lower
    brand_outside = any(b in outside for b in BRANDS) and not any(b in reg_label for b in BRANDS)
    letters_host = [c for c in reg_label if c.isalpha()]

    try:
        has_port = parts.port is not None
    except ValueError:
        has_port = True

    feats = {
        "url_len": n,
        "host_len": len(host),
        "path_len": len(path),
        "query_len": len(query),
        "fragment_len": len(fragment),
        "n_dots_url": url.count("."),
        "n_hyphens_url": url.count("-"),
        "n_underscores_url": url.count("_"),
        "n_slashes_url": url.count("/"),
        "n_qmarks": url.count("?"),
        "n_equals": url.count("="),
        "n_at": url.count("@"),
        "n_ampersands": url.count("&"),
        "n_tildes": url.count("~"),
        "n_percent": url.count("%"),
        "n_plus": url.count("+"),
        "digit_ratio_url": digits / n,
        "letter_ratio_url": letters / n,
        "special_ratio_url": specials / n,
        "url_entropy": shannon_entropy(url),
        "n_dots_host": host.count("."),
        "n_hyphens_host": host.count("-"),
        "n_digits_host": sum(c.isdigit() for c in host),
        "host_entropy": shannon_entropy(host),
        "n_subdomain_levels": len([s for s in subdomain.split(".") if s]),
        "tld_len": len(tld),
        "is_ip_host": int(ip_host),
        "has_port": int(has_port),
        "is_https": int(parts.scheme.lower() == "https"),
        "has_punycode": int("xn--" in host),
        "path_depth": len([p for p in path.split("/") if p]),
        "has_double_slash_in_path": int("//" in path),
        "has_web_extension": int(path.lower().endswith(_WEB_EXTENSIONS)),
        "has_risky_extension": int(path.lower().endswith(_RISKY_EXTENSIONS)),
        "n_query_params": len([q for q in query.split("&") if q]),
        "n_suspicious_keywords": sum(k in lower for k in SUSPICIOUS_KEYWORDS),
        "brand_outside_reg_domain": int(brand_outside),
        "is_shortener": int(reg_domain in SHORTENERS or host in SHORTENERS),
        "tld_is_risky": int(tld in RISKY_TLDS),
        "tld_is_common": int(tld in COMMON_TLDS),
        "longest_token_len": max((len(t) for t in tokens), default=0),
        "mean_token_len": (sum(len(t) for t in tokens) / len(tokens)) if tokens else 0.0,
        "max_digit_run": _max_run(url, str.isdigit),
        "max_consonant_run_host": _max_run(host, lambda c: c.isalpha() and c not in _VOWELS),
        "reg_domain_len": len(reg_label),
        "reg_domain_n_digits": sum(c.isdigit() for c in reg_label),
        "reg_domain_n_hyphens": reg_label.count("-"),
        "reg_domain_vowel_ratio": (sum(c in _VOWELS for c in letters_host) / len(letters_host)) if letters_host else 0.0,
        "has_hex_or_encoded": int(bool(re.search(r"%[0-9a-fA-F]{2}|0x[0-9a-fA-F]{4,}", url))),
        "has_email_in_url": int(bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", path + "?" + query))),
    }
    return {k: float(feats[k]) for k in LEXICAL_FEATURES}


# --------------------------------------------------------------------------- domain level
# Features of the *site domain only*. Scheme, "www.", subdomains, path and query are
# deliberately excluded: in the training data (PhiUSIIL) every legitimate URL is a
# bare https://www.<domain> homepage, so any of those would let the model learn the
# collection format instead of phishing. See README "Dataset artifacts".
DOMAIN_FEATURES: tuple[str, ...] = (
    "d_len", "d_label_len", "d_n_labels", "d_n_hyphens", "d_n_digits", "d_digit_ratio",
    "d_entropy", "d_vowel_ratio", "d_max_consonant_run", "d_max_digit_run",
    "d_max_repeat_run", "d_n_tokens", "d_longest_token", "d_digit_letter_switches",
    "d_tld_len", "d_tld_is_risky", "d_tld_is_common", "d_tld_is_cc", "d_suffix_n_parts",
    "d_on_private_suffix", "d_has_punycode", "d_is_ip", "d_brand_contained",
    "d_brand_with_extra", "d_n_suspicious_keywords", "d_is_shortener",
)


def extract_domain(raw_url: str) -> dict[str, float]:
    url = normalize_url(raw_url)
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    is_ip = _is_ip(host)
    if is_ip:
        site, suffix, private = host, "", False
    else:
        _, site, suffix, _, private = domain_parts(host)
    label = site[: -len(suffix) - 1] if suffix and site.endswith("." + suffix) else site
    tld = suffix.rsplit(".", 1)[-1] if suffix else ""
    letters = [c for c in label if c.isalpha()]
    tokens = [t for t in re.split(r"[-.]", label) if t]
    switches = sum(1 for a, b in zip(label, label[1:]) if a.isdigit() != b.isdigit() and a.isalnum() and b.isalnum())
    brand_hits = [b for b in BRANDS if len(b) >= 4 and b in label]
    feats = {
        "d_len": len(site),
        "d_label_len": len(label),
        "d_n_labels": site.count(".") + 1,
        "d_n_hyphens": label.count("-"),
        "d_n_digits": sum(c.isdigit() for c in label),
        "d_digit_ratio": (sum(c.isdigit() for c in label) / len(label)) if label else 0.0,
        "d_entropy": shannon_entropy(label),
        "d_vowel_ratio": (sum(c in _VOWELS for c in letters) / len(letters)) if letters else 0.0,
        "d_max_consonant_run": _max_run(label, lambda c: c.isalpha() and c not in _VOWELS),
        "d_max_digit_run": _max_run(label, str.isdigit),
        "d_max_repeat_run": max((len(m.group(0)) for m in re.finditer(r"(.)\1*", label)), default=0),
        "d_n_tokens": len(tokens),
        "d_longest_token": max((len(t) for t in tokens), default=0),
        "d_digit_letter_switches": switches,
        "d_tld_len": len(tld),
        "d_tld_is_risky": int(tld in RISKY_TLDS),
        "d_tld_is_common": int(tld in COMMON_TLDS),
        "d_tld_is_cc": int(len(tld) == 2),
        "d_suffix_n_parts": len(suffix.split(".")) if suffix else 0,
        "d_on_private_suffix": int(private),
        "d_has_punycode": int("xn--" in site),
        "d_is_ip": int(is_ip),
        "d_brand_contained": int(bool(brand_hits)),
        "d_brand_with_extra": int(any(label != b for b in brand_hits)),
        "d_n_suspicious_keywords": sum(k in label for k in SUSPICIOUS_KEYWORDS),
        "d_is_shortener": int(site in SHORTENERS),
    }
    return {k: float(feats[k]) for k in DOMAIN_FEATURES}
