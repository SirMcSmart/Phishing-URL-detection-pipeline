PY := .venv/bin/python

.PHONY: venv data enrich train test test-network serve docker-build docker-run all

venv:
	python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt && .venv/bin/pip install --no-deps -e .

data:            ## download PhiUSIIL and print de-dup stats
	$(PY) -m phishdetect.dataset

enrich:          ## live WHOIS/RDAP + DNS lookups for the enrichment sample (cached, resumable)
	$(PY) -m phishdetect.enrich_dataset

train:           ## train naive baseline + lexical + full models, write outputs/metrics.json
	$(PY) -m phishdetect.train

test:
	$(PY) -m pytest -q

test-network:
	$(PY) -m pytest -q -m network

serve:
	.venv/bin/uvicorn phishdetect.api:app --host 127.0.0.1 --port 8000 --reload

docker-build:
	docker build -t phishing-url-detection:latest .

docker-run:
	docker run --rm -p 127.0.0.1:8000:8000 --read-only --tmpfs /tmp --cap-drop ALL \
	  --security-opt no-new-privileges phishing-url-detection:latest

all: data enrich train test
