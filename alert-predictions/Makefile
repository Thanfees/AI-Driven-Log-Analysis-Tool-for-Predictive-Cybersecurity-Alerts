SHELL := /bin/bash

# Activate virtualenv (use LOG_FORECAST_VENV env var or default path)
VENV_PATH ?= /home/hackgodx/Projects/RP/venv/bin/activate
ACTIVATE := source "$(VENV_PATH)"

.PHONY: help gen-synth train train-mac train-windows train-all calibrate infer realtime demo test

help:
	@echo "Targets:"
	@echo "  gen-synth     - generate 80k synthetic log with low anomaly rate"
	@echo ""
	@echo "  train         - train Linux pipeline (60s windows; 15m horizon; trends)"
	@echo "  train-mac     - train Mac pipeline"
	@echo "  train-windows - train Windows pipeline"
	@echo "  train-all     - train all OS pipelines"
	@echo ""
	@echo "  calibrate     - calibrate threshold to ~5 alerts/day with K=3"
	@echo "  infer         - run batch inference on the synthetic file (calibrated)"
	@echo "  realtime      - one-command realtime replay + inference (Linux)"
	@echo "  demo          - all of the above in sequence (Linux)"
	@echo "  test          - run unit tests"

RAW_LOG := raw_logs/linux/synth_80k_loanom.log
MODEL_DIR := models/linux/baseline_combined_w60s_h15m
PRED_OUT := outputs/linux/synth_80k_loanom.log_predictions_calibrated.csv
WINDOWS_LOG := raw_logs/windows/Windows.log
WINDOWS_CHUNK_DIR := raw_logs/windows/chunks
WINDOWS_SPLIT_PARTS := 20

gen-synth:
	$(ACTIVATE); python scripts/generate_synthetic_linux_log.py \
		--lines 80000 --anomaly-frac 0.0015 --out $(RAW_LOG)

# ─── Linux Pipeline ──────────────────────────────────────────────
train:
	$(ACTIVATE); python scripts/00_run_pipeline.py \
		--os linux \
		--raw-dir raw_logs/linux \
		--converted-dir data/linux/raw_csv \
		--window 60s \
		--horizon-min 15 \
		--use-trends \
		--target-precision 0.80 \
		--min-lines 5 \
		--k-confirm 3 \
		--recursive

# ─── Mac Pipeline ────────────────────────────────────────────────
train-mac:
	$(ACTIVATE); python scripts/00_run_pipeline.py \
		--os mac \
		--raw-dir raw_logs/mac \
		--converted-dir data/mac/raw_csv \
		--window 60s \
		--horizon-min 15 \
		--use-trends \
		--target-precision 0.80 \
		--min-lines 5 \
		--k-confirm 3

# ─── Windows Pipeline ────────────────────────────────────────────
train-windows:
	$(ACTIVATE); RAW_DIR=raw_logs/windows; \
	if [ -f "$(WINDOWS_LOG)" ]; then \
		./scripts/split_windows_log.sh "$(WINDOWS_LOG)" $(WINDOWS_SPLIT_PARTS); \
		RAW_DIR="$(WINDOWS_CHUNK_DIR)"; \
	fi; \
	python scripts/00_run_pipeline.py \
		--os windows \
		--raw-dir $$RAW_DIR \
		--converted-dir data/windows/raw_csv \
		--window 60s \
		--horizon-min 15 \
		--use-trends \
		--target-precision 0.92 \
		--min-lines 8 \
		--k-confirm 3 \
		--recursive

# ─── All Platforms ───────────────────────────────────────────────
train-all: train train-mac train-windows

calibrate:
	$(ACTIVATE); python scripts/calibrate_threshold.py \
		--model-dir $(MODEL_DIR) \
		--k-confirm 3 \
		--target-alerts-per-day 5

infer:
	$(ACTIVATE); python src/linux/pipeline/07_infer_baseline.py \
		--input data/linux/labeled/$(notdir $(RAW_LOG))_windowz_labeled_trends.csv \
		--model-dir $(MODEL_DIR) \
		--output $(PRED_OUT) \
		--min-lines 5 \
		--k-confirm 3

realtime:
	$(ACTIVATE); bash scripts/realtime_runner.sh \
		-l $(RAW_LOG) \
		-m $(MODEL_DIR) \
		-o outputs/linux/realtime_demo.csv \
		-w 60 -k 3 -r 50

test:
	$(ACTIVATE); pytest tests/ -v

demo: gen-synth train calibrate infer realtime
