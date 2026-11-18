"""Live WHOIS (RDAP) + DNS enrichment of a class-balanced sample of the dataset.

Picks ``enrichment.sample_size`` URLs, at most one per registered domain, half
from each class, and looks each one up. Results go to an append-only JSONL
cache, so an interrupted run resumes where it stopped and re-runs cost nothing.

Usage:  python -m phishdetect.enrich_dataset [--sample-size N]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import threading
import time
from datetime import datetime, timezone

import pandas as pd

from . import dns_features
from .config import load_config, resolve
from .dataset import load_urls
from .features import ENRICHED_FEATURES, enrich, host_and_domain, looks_negative_or_failed, network_ok

log = logging.getLogger(__name__)


def cache_path(cfg):
    return resolve(cfg["data"]["processed_dir"]) / "enrichment_cache.jsonl"


def read_cache(cfg) -> dict[str, dict]:
    path = cache_path(cfg)
    out: dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                out[rec["url"]] = rec
    return out


def sample_urls(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    one_per_domain = df.sample(frac=1.0, random_state=seed).drop_duplicates("reg_domain")
    per_class = n // 2
    parts = []
    for label in (0, 1):
        pool = one_per_domain[one_per_domain["is_phishing"] == label]
        parts.append(pool.sample(n=min(per_class, len(pool)), random_state=seed))
    # Interleave the classes so transient network trouble affects both evenly.
    out = pd.concat(parts)
    out["_order"] = out.groupby("is_phishing").cumcount()
    return out.sort_values(["_order", "is_phishing"]).drop(columns="_order").reset_index(drop=True)


def _positive_score(rec: dict) -> int:
    """How much real-world presence a record shows (registered, resolves, has records)."""
    st, f = rec["status"], rec["features"]
    return (int(st["whois"].startswith("ok")) + int(f.get("dns_resolves") == 1)
            + int((f.get("n_ns_records") or 0) > 0))


def confirm_negatives(cfg: dict, urls: list[str], cache: dict[str, dict], workers: int = 2) -> int:
    """Re-query records with negative/failed outcomes at low concurrency.

    A flaky network can turn a live domain into "no address" or "not found", and it
    never does the reverse. So when the re-check shows *more* presence than the
    first answer, the re-check wins. Returns how many records changed.
    """
    ecfg = cfg["enrichment"]
    resolver = dns_features.make_resolver(ecfg["timeout_seconds"], ecfg.get("nameservers"))
    todo = [u for u in urls if u in cache and looks_negative_or_failed(cache[u]["status"])
            and not cache[u].get("confirmed")]
    log.info("confirm pass: re-checking %d negative/failed records", len(todo))
    changed = 0
    path = cache_path(cfg)

    def recheck(url):
        while not network_ok(resolver, max_age=0):
            time.sleep(15)
        feats, status = enrich(url, timeout=ecfg["timeout_seconds"] * 2, resolver=resolver)
        return url, feats, status

    with open(path, "a", encoding="utf-8") as fh, cf.ThreadPoolExecutor(workers) as pool:
        for i, (url, feats, status) in enumerate(pool.map(recheck, todo), 1):
            old = cache[url]
            new = {**old, "status": status, "features": {k: (None if v != v else v) for k, v in feats.items()},
                   "queried_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "confirmed": True}
            if _positive_score(new) > _positive_score(old):
                changed += 1
                rec = new
            else:
                rec = {**old, "confirmed": True}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            cache[url] = rec
            if i % 200 == 0:
                log.info("confirmed %d/%d (changed so far: %d)", i, len(todo), changed)
    log.info("confirm pass done: %d of %d records changed to a more-present answer", changed, len(todo))
    return changed


def run(cfg: dict, sample_size: int | None = None, confirm: bool = True) -> pd.DataFrame:
    ecfg = cfg["enrichment"]
    sample_size = sample_size or ecfg["sample_size"]
    df = load_urls(cfg)
    sample = sample_urls(df, sample_size, cfg["seed"])
    cache = read_cache(cfg)
    # Records whose lookup ended in a transient error are queried again; the newest
    # record for a URL wins when the cache is read back.
    retry = {u for u, rec in cache.items() if any(v.startswith("error") for v in rec["status"].values())}
    todo = sample[~sample["url"].isin(cache) | sample["url"].isin(retry)]
    log.info("sample=%d to_query=%d (of which %d retries of earlier errors)", len(sample), len(todo),
             int(todo["url"].isin(retry).sum()))

    path = cache_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    resolver = dns_features.make_resolver(ecfg["timeout_seconds"], ecfg.get("nameservers"))
    done = 0
    started = time.monotonic()

    def work(url: str) -> dict:
        waited = 0.0
        while True:
            feats, status = enrich(url, timeout=ecfg["timeout_seconds"], resolver=resolver)
            # A negative or failed answer is only trusted if the network is demonstrably up;
            # otherwise wait for it to come back and ask again (up to 30 minutes).
            if not looks_negative_or_failed(status) or network_ok(resolver, max_age=10.0):
                break
            if waited >= 1800:
                raise RuntimeError("network unavailable for 30 minutes; stopping (progress is cached)")
            log.warning("network canary failed; pausing before retrying %s", url)
            time.sleep(15)
            waited += 15
        site, icann, _ = host_and_domain(url)
        return {"url": url, "site_domain": site, "icann_domain": icann, "status": status,
                "features": {k: (None if v != v else v) for k, v in feats.items()},  # NaN -> null
                "queried_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    with open(path, "a", encoding="utf-8") as fh, cf.ThreadPoolExecutor(ecfg["workers"]) as pool:
        # as_completed, not map: one slow domain must not hold back every finished result.
        futures = [pool.submit(work, u) for u in todo["url"].tolist()]
        for fut in cf.as_completed(futures):
            rec = fut.result()
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                cache[rec["url"]] = rec
                done += 1
                if done % 200 == 0:
                    rate = done / (time.monotonic() - started)
                    log.info("enriched %d/%d (%.1f/s)", done, len(todo), rate)

    if confirm:
        confirm_negatives(cfg, sample["url"].tolist(), cache)

    rows = []
    for _, r in sample.iterrows():
        rec = cache[r["url"]]
        feats = {k: (float("nan") if rec["features"].get(k) is None else rec["features"][k])
                 for k in ENRICHED_FEATURES}
        rows.append({"url": r["url"], "is_phishing": r["is_phishing"], "reg_domain": r["reg_domain"],
                     "whois_status": rec["status"]["whois"], "dns_status": rec["status"]["dns"],
                     "queried_at": rec["queried_at"], **feats})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=None)
    ap.add_argument("--no-confirm", action="store_true", help="skip the re-check of negative answers")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("whois").setLevel(logging.CRITICAL)
    cfg = load_config()
    out = run(cfg, args.sample_size, confirm=not args.no_confirm)
    dest = resolve(cfg["data"]["processed_dir"]) / "enriched_sample.csv"
    out.to_csv(dest, index=False)
    summary = {
        "rows": len(out),
        "whois_status": out["whois_status"].str.split(":").str[0].value_counts().to_dict(),
        "dns_status": out["dns_status"].value_counts().to_dict(),
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
