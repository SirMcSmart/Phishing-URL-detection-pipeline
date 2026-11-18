"""Train and evaluate the gradient-boosting models.

* ``naive_full_url_baseline`` (reported, never served): lexical features of the
  whole URL. Shows the shortcut the dataset allows; see README.
* ``lexical``: domain-level lexical features, one row per site domain, whole dataset.
* ``full``:    domain lexical + WHOIS + DNS, trained on the live-enriched sample
               (run ``python -m phishdetect.enrich_dataset`` first).

All splits are grouped by registered domain. Every number written to
``outputs/metrics.json`` is computed here on held-out data; nothing is
hand-entered.

Usage:  python -m phishdetect.train [--skip-full]
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit

from .config import load_config, resolve
from .dataset import load_urls
from .features import ENRICHED_FEATURES, FULL_FEATURES
from .url_features import DOMAIN_FEATURES, LEXICAL_FEATURES, extract_domain, extract_lexical

log = logging.getLogger(__name__)
MODEL_VERSION = "1.0.0"


# ----------------------------------------------------------------------------- helpers
def grouped_split(df: pd.DataFrame, test_size: float, val_size: float, seed: int):
    """Train/val/test index arrays with no registered domain shared between them."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trval_idx, test_idx = next(gss.split(df, groups=df["reg_domain"]))
    trval = df.iloc[trval_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed + 1)
    tr_rel, val_rel = next(gss2.split(trval, groups=trval["reg_domain"]))
    return trval_idx[tr_rel], trval_idx[val_rel], test_idx


def make_model(mcfg: dict, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=mcfg["learning_rate"], max_iter=mcfg["max_iter"],
        max_leaf_nodes=mcfg["max_leaf_nodes"], min_samples_leaf=mcfg["min_samples_leaf"],
        l2_regularization=mcfg["l2_regularization"], early_stopping=mcfg["early_stopping"],
        validation_fraction=mcfg["validation_fraction"], n_iter_no_change=mcfg["n_iter_no_change"],
        random_state=seed,
    )


def pick_threshold(y: np.ndarray, p: np.ndarray, strategy) -> float:
    if strategy == "f1":
        prec, rec, thr = precision_recall_curve(y, p)
        f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
        return float(thr[int(np.argmax(f1))])
    max_fpr = float(strategy)
    fpr, _, thr = roc_curve(y, p)
    ok = np.where(fpr <= max_fpr)[0]
    return float(min(thr[ok[-1]], 1.0))


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)), "positives": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "pr_auc": float(average_precision_score(y, p)) if both else None,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def lexical_matrix(urls: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([extract_lexical(u) for u in urls], columns=list(LEXICAL_FEATURES))


def domain_matrix(urls: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([extract_domain(u) for u in urls], columns=list(DOMAIN_FEATURES))


def domain_level(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per site domain. Domains carrying both labels are dropped."""
    n_labels = df.groupby("reg_domain")["is_phishing"].nunique()
    conflicted = n_labels[n_labels > 1].index
    out = (df[~df["reg_domain"].isin(conflicted)]
           .drop_duplicates("reg_domain").reset_index(drop=True))
    return out, {"domains": len(out), "domains_with_conflicting_labels_dropped": int(len(conflicted)),
                 "phishing": int(out["is_phishing"].sum()),
                 "legitimate": int(len(out) - out["is_phishing"].sum())}


def fit_eval(X: pd.DataFrame, y: np.ndarray, tr, va, te, mcfg, seed, features):
    model = make_model(mcfg, seed).fit(X.iloc[tr][list(features)], y[tr])
    p_val = model.predict_proba(X.iloc[va][list(features)])[:, 1]
    thr = pick_threshold(y[va], p_val, mcfg["threshold_strategy"])
    p_test = model.predict_proba(X.iloc[te][list(features)])[:, 1]
    return model, thr, p_test, {"threshold": thr, "n_iter": int(model.n_iter_),
                                "val": evaluate(y[va], p_val, thr), "test": evaluate(y[te], p_test, thr)}


def top_importances(model, X, y, features, seed, n_repeats=3, sample=5000) -> list[dict]:
    idx = np.random.default_rng(seed).choice(len(X), size=min(sample, len(X)), replace=False)
    r = permutation_importance(model, X.iloc[idx][list(features)], y[idx], scoring="average_precision",
                               n_repeats=n_repeats, random_state=seed, n_jobs=1)
    order = np.argsort(-r.importances_mean)
    return [{"feature": features[i], "pr_auc_drop": round(float(r.importances_mean[i]), 5)}
            for i in order[:15]]


def save_model(path, model, features, threshold, name, extra: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(features), "threshold": threshold, "name": name,
                 "version": MODEL_VERSION, "sklearn_version": sklearn.__version__,
                 "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **extra}, path)


# ----------------------------------------------------------------------------- stages
def train_naive_baseline(cfg: dict) -> dict:
    seed, mcfg, scfg = cfg["seed"], cfg["model"], cfg["split"]
    df = load_urls(cfg)
    stats = df.attrs["stats"]
    log.info("dataset: %s", stats)
    cache = resolve(cfg["data"]["processed_dir"]) / "lexical_features.pkl"
    if cache.exists() and len(pd.read_pickle(cache)) == len(df):
        X = pd.read_pickle(cache)
    else:
        log.info("extracting lexical features for %d URLs", len(df))
        X = lexical_matrix(df["url"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        X.to_pickle(cache)
    y = df["is_phishing"].to_numpy()
    tr, va, te = grouped_split(df, scfg["test_size"], scfg["val_size"], seed)
    assert not set(df["reg_domain"].iloc[te]) & set(df["reg_domain"].iloc[tr])

    model, thr, p_test, res = fit_eval(X, y, tr, va, te, mcfg, seed, LEXICAL_FEATURES)
    log.info("naive full-URL baseline test: %s", res["test"])

    # Dataset-artifact check: in PhiUSIIL most legitimate URLs are bare homepages,
    # so "has a path" alone separates classes well. Report performance on the
    # test URLs that have no path/query, where that shortcut is unavailable.
    Xte = X.iloc[te]
    bare = ((Xte["path_depth"] == 0) & (Xte["query_len"] == 0)).to_numpy()
    res["test_bare_homepage_urls_only"] = evaluate(y[te][bare], p_test[bare], thr)
    res["split_sizes"] = {"train": len(tr), "val": len(va), "test": len(te)}
    res["permutation_importance_top15"] = top_importances(model, X.iloc[te], y[te], LEXICAL_FEATURES, seed)

    artifact = {}
    for col in ("is_https", "path_depth", "query_len", "url_len", "n_subdomain_levels", "tld_is_common"):
        g = X.groupby(df["is_phishing"])[col]
        artifact[col] = {"legit_mean": round(float(g.mean()[0]), 4), "phish_mean": round(float(g.mean()[1]), 4)}
    www = df["url"].str.lower().str.contains(r"^(?:https?://)?www\.", regex=True)
    artifact["share_host_starts_www"] = {
        lbl: round(float(www[df["is_phishing"] == v].mean()), 4) for lbl, v in (("legit", 0), ("phish", 1))}
    artifact["share_https"] = {
        lbl: round(float(X["is_https"][df["is_phishing"] == v].mean()), 4) for lbl, v in (("legit", 0), ("phish", 1))}
    artifact["share_with_no_path_or_query"] = {
        lbl: round(float(((X["path_depth"] == 0) & (X["query_len"] == 0))[df["is_phishing"] == v].mean()), 4)
        for lbl, v in (("legit", 0), ("phish", 1))}

    # Not saved: this model is a demonstration of shortcut learning, not a product.
    return {"dataset": stats, "naive_full_url_baseline": res, "dataset_artifacts": artifact}


def train_lexical(cfg: dict) -> dict:
    seed, mcfg, scfg = cfg["seed"], cfg["model"], cfg["split"]
    dom, dstats = domain_level(load_urls(cfg))
    log.info("domain-level dataset: %s", dstats)
    X = domain_matrix(dom["url"])
    y = dom["is_phishing"].to_numpy()
    tr, va, te = grouped_split(dom, scfg["test_size"], scfg["val_size"], seed)
    model, thr, p_test, res = fit_eval(X, y, tr, va, te, mcfg, seed, DOMAIN_FEATURES)
    log.info("domain lexical test: %s", res["test"])
    res["dataset"] = dstats
    res["split_sizes"] = {"train": len(tr), "val": len(va), "test": len(te)}
    res["permutation_importance_top15"] = top_importances(model, X.iloc[te], y[te], DOMAIN_FEATURES, seed)
    # Slices that matter operationally.
    Xte, yte = X.iloc[te], y[te]
    for name, mask in {"common_tld_only": Xte["d_tld_is_common"] == 1,
                       "private_suffix_hosts": Xte["d_on_private_suffix"] == 1}.items():
        m = mask.to_numpy()
        if m.sum():
            res[f"test_slice_{name}"] = evaluate(yte[m], p_test[m], thr)
    save_model(resolve(cfg["artifacts"]["models_dir"]) / "lexical.joblib", model, DOMAIN_FEATURES, thr,
               "lexical", {"metrics": res["test"]})
    return {"lexical": res}


def train_full(cfg: dict) -> dict | None:
    seed, mcfg, scfg = cfg["seed"], cfg["model"], cfg["split"]
    path = resolve(cfg["data"]["processed_dir"]) / "enriched_sample.csv"
    if not path.exists():
        log.warning("no %s; run `python -m phishdetect.enrich_dataset` first. Skipping full model.", path)
        return None
    es = pd.read_csv(path)
    X = pd.concat([domain_matrix(es["url"]), es[list(ENRICHED_FEATURES)].reset_index(drop=True)], axis=1)
    y = es["is_phishing"].to_numpy()
    tr, va, te = grouped_split(es, scfg["test_size"], scfg["val_size"], seed)

    out: dict = {"n_rows": len(es), "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
                 "queried_between": [es["queried_at"].min(), es["queried_at"].max()]}
    # Coverage and class-conditional availability: shows the "dead phishing domain" bias.
    by = es.groupby("is_phishing")
    out["lookup_coverage"] = {
        name: {
            "whois_found_rate": round(float((g["whois_found"] == 1).mean()), 4),
            "whois_not_found_rate": round(float((g["whois_found"] == 0).mean()), 4),
            "whois_error_rate": round(float(g["whois_status"].str.startswith("error").mean()), 4),
            "dns_resolves_rate": round(float((g["dns_resolves"] == 1).mean()), 4),
            "dns_error_rate": round(float(g["dns_status"].str.startswith("error").mean()), 4),
        } for name, g in (("legit", by.get_group(0)), ("phish", by.get_group(1)))
    }

    # Ablation on the same split: which feature groups carry the signal?
    groups = {"lexical_only": DOMAIN_FEATURES, "enrichment_only": ENRICHED_FEATURES, "full": FULL_FEATURES}
    full_model = full_thr = None
    for name, feats in groups.items():
        model, thr, _, res = fit_eval(X, y, tr, va, te, mcfg, seed, feats)
        out[name] = res
        log.info("%s test: %s", name, res["test"])
        if name == "full":
            full_model, full_thr = model, thr
            res["permutation_importance_top15"] = top_importances(model, X.iloc[te], y[te], FULL_FEATURES,
                                                                  seed, n_repeats=5)
    # Same model, but scored only on rows whose domain still resolves and is registered:
    # removes the easy "phishing domain is already dead" cases.
    live = ((X["dns_resolves"] == 1) & (X["whois_found"] == 1)).to_numpy()
    te_live = te[live[te]]
    if len(np.unique(y[te_live])) == 2:
        p = full_model.predict_proba(X.iloc[te_live][list(FULL_FEATURES)])[:, 1]
        out["full_on_live_domains_only"] = evaluate(y[te_live], p, full_thr)

    save_model(resolve(cfg["artifacts"]["models_dir"]) / "full.joblib", full_model, FULL_FEATURES, full_thr,
               "full", {"metrics": out["full"]["test"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--skip-lexical", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    out_path = resolve(cfg["artifacts"]["outputs_dir"]) / "metrics.json"
    metrics = json.loads(out_path.read_text()) if out_path.exists() else {}
    metrics["environment"] = {"python": platform.python_version(), "sklearn": sklearn.__version__,
                              "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if not args.skip_lexical:
        metrics.update(train_naive_baseline(cfg))
        metrics.update(train_lexical(cfg))
    if not args.skip_full:
        full = train_full(cfg)
        if full is not None:
            metrics["full_model"] = full
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
