# Data

Nothing in `data/raw/` or `data/processed/` is committed. Recreate it with `make data enrich`.

## PhiUSIIL Phishing URL Dataset

- Source: UCI Machine Learning Repository, dataset 967 —
  https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
- Licence: **Creative Commons Attribution 4.0 International (CC BY 4.0)** (stated on the UCI page, checked 2026-11-17)
- Size: 235,795 rows (134,850 legitimate, 100,945 phishing), as published.
- Columns used: `URL` and `label` only (`label` 1 = legitimate, 0 = phishing; this project flips it so
  `is_phishing = 1`). The HTML/page-derived columns are ignored because this service never fetches pages.

Citation:

> Prasad, A., & Chandra, S. (2024). PhiUSIIL: A diverse security profile empowered phishing URL
> detection framework based on similarity index and incremental learning. *Computers & Security*, 136,
> 103545. https://doi.org/10.1016/j.cose.2023.103545

## Live enrichment (`data/processed/enrichment_cache.jsonl`)

WHOIS/RDAP and DNS answers collected by `python -m phishdetect.enrich_dataset` for a class-balanced
sample of site domains. Each record stores the UTC time it was queried. These answers describe the
domains *at query time*, not at the time the dataset was collected (see README, "Limitations").

- RDAP servers come from the IANA bootstrap file https://data.iana.org/rdap/dns.json
- DNS is queried through 1.1.1.1 and 8.8.8.8 (configurable in `config.yaml`)
