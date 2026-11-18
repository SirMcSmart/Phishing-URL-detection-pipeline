"""Write the README "Results" section from outputs/metrics.json.

Numbers in the README are generated, never typed by hand. Run after training:
    python -m phishdetect.report
"""
from __future__ import annotations

import json
import re

from .config import resolve

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
A_START, A_END = "<!-- ARTIFACTS:START -->", "<!-- ARTIFACTS:END -->"


def render_artifacts(metrics: dict) -> str:
    a = metrics["dataset_artifacts"]
    rows = [("Scheme is `https`", a["share_https"]), ("Host starts with `www.`", a["share_host_starts_www"]),
            ("No path and no query (bare homepage)", a["share_with_no_path_or_query"])]
    return "\n".join(["| Property (share of URLs) | Legitimate | Phishing |", "|---|---:|---:|"] +
                     [f"| {n} | {v['legit']:.1%} | {v['phish']:.1%} |" for n, v in rows])


def _replace(text: str, start: str, end: str, body: str) -> str:
    return re.sub(re.escape(start) + r".*?" + re.escape(end), lambda _: f"{start}\n{body}\n{end}", text, flags=re.S)


def _row(name: str, m: dict, note: str = "") -> str:
    def f(x, nd=3):
        return "n/a" if x is None else f"{x:.{nd}f}"
    return (f"| {name} | {m['n']:,} | {f(m['roc_auc'])} | {f(m['pr_auc'])} | {f(m['precision'])} | "
            f"{f(m['recall'])} | {f(m['f1'])} | {f(m['fpr'], 4)} | {note} |")


def render(metrics: dict) -> str:
    out = [f"_Generated from `outputs/metrics.json` ({metrics['environment']['generated_at']}, "
           f"scikit-learn {metrics['environment']['sklearn']}). All splits are grouped by site domain._", ""]
    ds = metrics.get("dataset", {})
    lex = metrics.get("lexical", {})
    if ds:
        out += [f"**Data:** {ds['rows_raw']:,} PhiUSIIL rows → {ds['rows_final']:,} unique valid URLs "
                f"({ds['phishing']:,} phishing / {ds['legitimate']:,} legitimate)."]
    if lex.get("dataset"):
        d = lex["dataset"]
        out += [f"Domain-level dataset: {d['domains']:,} site domains ({d['phishing']:,} phishing / "
                f"{d['legitimate']:,} legitimate; {d['domains_with_conflicting_labels_dropped']} domains "
                f"with both labels dropped).", ""]

    hdr = ["| Model (held-out test set) | n | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR | Note |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    out += ["### 1. Whole dataset", ""] + hdr
    if "naive_full_url_baseline" in metrics:
        out.append(_row("Naive full-URL lexical (not served)", metrics["naive_full_url_baseline"]["test"],
                        "exploits dataset artifacts"))
    if lex:
        out.append(_row("**Domain lexical** (`lexical`)", lex["test"], "served fallback"))
        for k, label in (("test_slice_common_tld_only", "↳ slice: .com/.org/.net/.edu/.gov"),
                         ("test_slice_private_suffix_hosts", "↳ slice: shared-hosting tenants")):
            if k in lex:
                out.append(_row(label, lex[k]))
    out.append("")

    full = metrics.get("full_model")
    if full:
        q = full["queried_between"]
        out += ["### 2. Live-enriched sample (lexical vs WHOIS/DNS ablation)", "",
                f"{full['n_rows']:,} site domains, one URL each, class-balanced; WHOIS/RDAP and DNS queried "
                f"{q[0]} → {q[1]} (UTC). Same grouped split for every row below.", ""] + hdr
        out.append(_row("Domain lexical only", full["lexical_only"]["test"]))
        out.append(_row("WHOIS + DNS only", full["enrichment_only"]["test"]))
        out.append(_row("**Full** (`full`)", full["full"]["test"], "served when lookups succeed"))
        if "full_on_live_domains_only" in full:
            out.append(_row("↳ full, live domains only", full["full_on_live_domains_only"],
                            "registered + resolving"))
        out += ["", "**Lookup coverage by class** (why the WHOIS/DNS numbers need care):", "",
                "| Class | WHOIS record found | Registry says not registered | WHOIS error | DNS resolves | DNS error |",
                "|---|---:|---:|---:|---:|---:|"]
        for cls, c in full["lookup_coverage"].items():
            out.append(f"| {cls} | {c['whois_found_rate']:.1%} | {c['whois_not_found_rate']:.1%} | "
                       f"{c['whois_error_rate']:.1%} | {c['dns_resolves_rate']:.1%} | {c['dns_error_rate']:.1%} |")
        out.append("")
        imp = full["full"].get("permutation_importance_top15", [])[:8]
        if imp:
            out += ["Top features of the full model (permutation importance, drop in PR-AUC on test): " +
                    ", ".join(f"`{i['feature']}` ({i['pr_auc_drop']:.3f})" for i in imp) + ".", ""]

    if lex.get("permutation_importance_top15"):
        out += ["Top features of the domain-lexical model: " + ", ".join(
            f"`{i['feature']}` ({i['pr_auc_drop']:.3f})" for i in lex["permutation_importance_top15"][:8]) + ".", ""]
    nb = metrics.get("naive_full_url_baseline", {}).get("permutation_importance_top15", [])[:3]
    if nb:
        out += ["Top features of the naive full-URL model: " + ", ".join(
            f"`{i['feature']}` ({i['pr_auc_drop']:.3f})" for i in nb) +
                ". These are the collection artifacts described above.", ""]
    return "\n".join(out)


def main() -> None:
    metrics = json.loads(resolve("outputs/metrics.json").read_text())
    readme = resolve("README.md")
    text = readme.read_text()
    text = _replace(text, START, END, render(metrics))
    if "dataset_artifacts" in metrics:
        text = _replace(text, A_START, A_END, render_artifacts(metrics))
    readme.write_text(text)
    print("README results section updated")


if __name__ == "__main__":
    main()
