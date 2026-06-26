PYTHON ?= python3
PIP_REQUIREMENTS ?= requirements-dev.txt
PIP_CONSTRAINTS ?= requirements.lock
PYTEST_TIMEOUT ?= 300
HYGIENE_FLAGS ?=
LINT_TARGETS := evaluation/benchmark.py neurosight/utils tests/test_benchmark.py tests/test_reproducibility_smoke.py tests/test_training_smoke.py tests/test_gradient_clipping.py scripts/check_repo_hygiene.py scripts/check_python_version.py scripts/import_smoke.py scripts/prepare_adni_cognitive.py tests/test_prepare_adni_cognitive.py

.PHONY: check-python-runtime install dev test test-fast test-slow test-benchmark lint format hygiene import-smoke train demo smoke-api smoke-backend api-contract-check mlflow-registry dvc-provenance otel-probe fhir-export dicomweb-manifest langgraph-workflow drift-monitor onnx-export supply-chain-audit supply-chain-audit-local ai-safety-eval model-card-check quality-gate github-readiness portfolio-check deploy generate-synthetic seed-kg docker-up docker-down quality benchmark-smoke safety frontend-check verify

check-python-runtime:
	@if ! command -v "$(PYTHON)" >/dev/null 2>&1; then \
		echo "Python interpreter not found: $(PYTHON). Install Python 3.11 or run make with PYTHON=/path/to/python3.11."; \
		exit 1; \
	fi
	$(PYTHON) scripts/check_python_version.py

install: check-python-runtime
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r $(PIP_REQUIREMENTS) -c $(PIP_CONSTRAINTS)
	$(PYTHON) -m pip install flwr==1.8.0 --no-deps

dev: check-python-runtime
	$(PYTHON) -m pip install -r $(PIP_REQUIREMENTS) -c $(PIP_CONSTRAINTS)

test: check-python-runtime
	APP_ENV=test $(PYTHON) -m pytest tests/ -v --tb=short --timeout=$(PYTEST_TIMEOUT)

test-fast: check-python-runtime
	APP_ENV=test $(PYTHON) -m pytest tests/ -v -m "not slow and not benchmark and not safety" --tb=short --timeout=$(PYTEST_TIMEOUT)

test-slow: check-python-runtime
	APP_ENV=test $(PYTHON) -m pytest tests/ -v -m "slow" --tb=short --timeout=$(PYTEST_TIMEOUT)

test-benchmark: check-python-runtime
	APP_ENV=test $(PYTHON) -m pytest tests/test_benchmark.py -v -m "benchmark" --tb=short --timeout=$(PYTEST_TIMEOUT)

lint: check-python-runtime
	$(PYTHON) -m ruff check $(LINT_TARGETS) --ignore E501
	$(PYTHON) -m mypy evaluation/benchmark.py neurosight/utils/seed.py --ignore-missing-imports

format:
	$(PYTHON) -m ruff format .

hygiene: check-python-runtime
	@if [ -f scripts/check_repo_hygiene.py ]; then \
		$(PYTHON) scripts/check_repo_hygiene.py $(HYGIENE_FLAGS); \
	else \
		echo "scripts/check_repo_hygiene.py not found; skipping repository hygiene."; \
	fi

import-smoke: check-python-runtime
	APP_ENV=test PYTHONPATH=. $(PYTHON) scripts/import_smoke.py

train:
	$(PYTHON) scripts/train.py

demo:
	$(PYTHON) app.py

smoke-api:
	APP_ENV=test $(PYTHON) scripts/smoke_backend.py --skip-uploads

smoke-backend:
	APP_ENV=test $(PYTHON) scripts/smoke_backend.py

api-contract-check:
	$(PYTHON) scripts/api_contract_check.py --strict

mlflow-registry:
	$(PYTHON) scripts/mlflow_registry.py

dvc-provenance:
	$(PYTHON) scripts/dvc_provenance.py

otel-probe:
	$(PYTHON) scripts/otel_probe.py

fhir-export:
	$(PYTHON) scripts/fhir_export.py

dicomweb-manifest:
	$(PYTHON) scripts/dicomweb_manifest.py

langgraph-workflow:
	$(PYTHON) scripts/langgraph_workflow.py

drift-monitor:
	$(PYTHON) scripts/drift_monitor.py

onnx-export:
	$(PYTHON) scripts/onnx_export.py

supply-chain-audit:
	$(PYTHON) scripts/supply_chain_audit.py --strict-on critical

supply-chain-audit-local:
	$(PYTHON) scripts/supply_chain_audit.py --include-local-secrets

ai-safety-eval:
	$(PYTHON) scripts/ai_safety_eval.py --strict

model-card-check: check-python-runtime
	$(PYTHON) scripts/model_card_check.py --strict --stdout

quality-gate: check-python-runtime
	$(PYTHON) scripts/quality_gate.py --strict --stdout

github-readiness:
	$(PYTHON) scripts/github_readiness.py --strict

portfolio-check:
	$(PYTHON) scripts/portfolio_check.py --strict

deploy:
	gradio deploy

generate-synthetic:
	$(PYTHON) neurosight/scripts/download_data.py

seed-kg:
	$(PYTHON) neurosight/scripts/seed_kg.py

docker-up:
	docker compose up -d

docker-down:
	docker compose down

quality: check-python-runtime
	$(PYTHON) -m ruff check $(LINT_TARGETS) --ignore E501
	$(PYTHON) -m mypy evaluation/benchmark.py neurosight/utils/seed.py --ignore-missing-imports
	@if [ -f scripts/check_repo_hygiene.py ]; then $(PYTHON) scripts/check_repo_hygiene.py --allow-dev-caches $(HYGIENE_FLAGS); fi
	$(PYTHON) scripts/model_card_check.py --strict --stdout
	$(PYTHON) scripts/quality_gate.py --strict --stdout
	$(PYTHON) scripts/supply_chain_audit.py --strict-on critical

benchmark-smoke: check-python-runtime
	APP_ENV=test $(PYTHON) scripts/run_benchmark.py --mode smoke

safety: check-python-runtime
	APP_ENV=test $(PYTHON) -m pytest tests/test_safety_redteam.py -v -m "safety" --tb=short --timeout=$(PYTEST_TIMEOUT)

frontend-check:
	@set -e; \
	if [ -d frontend/node_modules ]; then \
		npm --prefix frontend run type-check; \
		if [ "$$CI" = "true" ]; then \
			npm --prefix frontend run build; \
		else \
			echo "CI is not true; skipping optional frontend build."; \
		fi; \
	else \
		echo "frontend/node_modules not found; skipping frontend type-check/build. Run npm --prefix frontend ci to enable."; \
	fi

verify: check-python-runtime quality import-smoke test-fast safety benchmark-smoke frontend-check
