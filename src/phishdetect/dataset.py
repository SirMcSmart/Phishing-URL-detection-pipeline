"""Download and prepare the PhiUSIIL dataset.

Only two columns are used: the raw URL and its label. PhiUSIIL also ships
features computed from each page's HTML; those are ignored on purpose, because
this project scores a URL *without* fetching the page.

PhiUSIIL label convention: 1 = legitimate, 0 = phishing. This project flips it
so that ``is_phishing = 1`` is the positive class.

Usage:  python -m phishdetect.dataset
"""
from __future__ import annotations

import io
import json
import logging
import zipfile

import pandas as pd
import requests

from .config import load_config, resolve
from .url_features import InvalidURLError, normalize_url, registered_domain

log = logging.getLogger(__name__)

# Counts published on the UCI dataset page; used as a sanity check of the download.
EXPECTED_ROWS = 235_795
EXPECTED_LEGIT = 134_850


def download(cfg: dict) -> None:
    raw_dir = resolve(cfg["data"]["raw_dir"])
    csv_path = raw_dir / cfg["data"]["csv_name"]
    if csv_path.exists() and len(pd.read_csv(csv_path, usecols=["URL"])) == EXPECTED_ROWS:
        log.info("dataset already present: %s", csv_path)
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", cfg["data"]["phiusiil_url"])
    resp = requests.get(cfg["data"]["phiusiil_url"], timeout=900)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        csv_path.write_bytes(zf.read(member))


def load_urls(cfg: dict) -> pd.DataFrame:
    """Return a de-duplicated frame with columns url, is_phishing, reg_domain."""
    csv_path = resolve(cfg["data"]["raw_dir"]) / cfg["data"]["csv_name"]
    df = pd.read_csv(csv_path, usecols=["URL", "label"], encoding="utf-8-sig")
    if len(df) != EXPECTED_ROWS or int((df["label"] == 1).sum()) != EXPECTED_LEGIT:
        raise RuntimeError(f"unexpected dataset shape {df.shape}; re-download with `python -m phishdetect.dataset`")
    df = df.rename(columns={"URL": "url"})
    df["is_phishing"] = (df["label"] == 0).astype(int)

    stats = {"rows_raw": len(df)}
    df["url"] = df["url"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["url"])
    stats["rows_after_exact_dedup"] = len(df)
    # A URL labelled both ways is unusable.
    conflicted = df.groupby("url")["is_phishing"].nunique()
    df = df[~df["url"].isin(conflicted[conflicted > 1].index)]

    def _safe_domain(u: str):
        try:
            normalize_url(u)
            return registered_domain(u)
        except InvalidURLError:
            return None

    df["reg_domain"] = df["url"].map(_safe_domain)
    stats["rows_invalid_url"] = int(df["reg_domain"].isna().sum())
    df = df.dropna(subset=["reg_domain"]).reset_index(drop=True)
    stats["rows_final"] = len(df)
    stats["phishing"] = int(df["is_phishing"].sum())
    stats["legitimate"] = int(len(df) - df["is_phishing"].sum())
    stats["unique_registered_domains"] = int(df["reg_domain"].nunique())
    df.attrs["stats"] = stats
    return df[["url", "is_phishing", "reg_domain"]]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    download(cfg)
    df = load_urls(cfg)
    print(json.dumps(df.attrs["stats"], indent=2))


if __name__ == "__main__":
    main()
