# Log Forecast — Multi-OS Early Warning System

A machine learning pipeline for predicting system anomalies before they occur.
Supports **Linux**, **macOS**, and **Windows** log formats.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Multi-OS Pipeline Flow                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📁 Raw Logs          OS-specific 01_convert_log_to_csv.py       │
│  (Linux/Mac/Win)  ──▶  Convert to structured CSV                 │
│                                                                  │
│       ▼                                                          │
│  📊 CSV Data          OS-specific 02_windowize.py                │
│                  ──▶  Create windows + OS keyword features       │
│                                                                  │
│       ▼                                                          │
│  🏷️ Windows           OS-specific 03_label_windows.py            │
│                  ──▶  Apply OS-specific anomaly labels           │
│                                                                  │
│       ▼                                                          │
│  📈 Labeled           04_add_trends.py      (shared)             │
│                  ──▶  Add rolling trend features                 │
│                                                                  │
│       ▼                                                          │
│  🎯 Features          05_make_future_labels.py (shared)          │
│                  ──▶  Create prediction targets                  │
│                                                                  │
│       ▼                                                          │
│  🤖 Training          06_train_baseline.py (shared)              │
│                  ──▶  Train Logistic Regression or GRU model     │
│                                                                  │
│       ▼                                                          │
│  🔮 Inference         07_infer_baseline.py (shared)              │
│                  ──▶  Batch or realtime predictions              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Activate environment
source "/home/hackgodx/Projects/RP/venv/bin/activate"

# Run full demo (Linux)
make demo

# Train all platforms
make train-all
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Unified Pipeline Runner

```bash
# Linux
python scripts/00_run_pipeline.py --os linux --raw-dir raw_logs/linux \
    --window 60s --horizon-min 15 --use-trends --target-precision 0.80

# macOS
python scripts/00_run_pipeline.py --os mac --raw-dir raw_logs/mac \
    --window 60s --horizon-min 15 --use-trends --target-precision 0.80

# Windows
python scripts/00_run_pipeline.py --os windows --raw-dir raw_logs/windows \
    --window 60s --horizon-min 15 --use-trends --target-precision 0.80
```

### Makefile Targets

```bash
make train           # Linux pipeline
make train-mac       # macOS pipeline
make train-windows   # Windows pipeline
make train-all       # All platforms
make calibrate       # Calibrate threshold
make infer           # Batch inference (Linux)
make realtime        # Realtime demo (Linux)
make test            # Run tests
```

### Calibrate Threshold

```bash
python scripts/calibrate_threshold.py \
    --model-dir models/linux/baseline_combined_w60s_h15m \
    --k-confirm 3 \
    --target-alerts-per-day 5
```

### Run Inference

```bash
python src/linux/pipeline/07_infer_baseline.py \
    --input data/linux/labeled/synth.log_windowz_labeled_trends.csv \
    --model-dir models/linux/baseline_combined_w60s_h15m \
    --output outputs/linux/predictions.csv \
    --min-lines 5 \
    --k-confirm 3
```

## Project Structure

```
alert-predictions/
├── src/
│   ├── common/                  # Shared utilities
│   │   └── constants.py         # Shared defaults & logging
│   ├── linux/                   # Linux pipeline
│   │   ├── common/constants.py  # Syslog regex, Linux keywords
│   │   ├── pipeline/            # Steps 01-08
│   │   └── realtime/            # Real-time monitoring
│   ├── mac/                     # macOS pipeline
│   │   ├── common/constants.py  # Mac syslog regex, Mac keywords
│   │   └── pipeline/            # Steps 01-03 (OS-specific)
│   └── windows/                 # Windows pipeline
│       ├── common/constants.py  # CBS/CSI regex, Windows keywords
│       └── pipeline/            # Steps 01-03 (OS-specific)
├── scripts/
│   ├── 00_run_pipeline.py       # Unified multi-OS runner
│   ├── 00_run_linux_pipeline.py # Legacy Linux-only runner
│   ├── calibrate_threshold.py
│   └── ...
├── raw_logs/                    # Sample raw log files (per-OS folders)
│   ├── linux/                   # Linux syslog
│   │   ├── linux.log
│   │   ├── synth_80k.log
│   │   ├── synth_80k_loanom.log
│   │   └── synthetic_60k.log
│   ├── mac/                     # macOS syslog
│   │   └── Mac.log
│   └── windows/                 # Windows CBS/CSI
│       └── windows20k.log
├── tests/
│   ├── test_constants.py        # Linux constants tests
│   ├── test_windows_constants.py
│   ├── test_mac_constants.py
│   ├── test_labeling.py
│   └── test_windowize.py
├── Makefile
├── requirements.txt
└── README.md
```

## Supported Log Formats

| OS | Format | Example |
|----|--------|---------|
| **Linux** | Syslog | `Jun  9 06:06:20 combo sshd[1234]: msg` |
| **macOS** | Syslog-style | `Jul  1 09:00:55 host kernel[0]: msg` |
| **Windows** | CBS/CSI | `2016-09-28 04:30:30, Info CBS msg` |

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--os` | Target OS: linux, mac, windows | — |
| `--window` | Window size (30s, 60s, 5min) | 60s |
| `--horizon-min` | Prediction horizon in minutes | 15 |
| `--target-precision` | Minimum precision for threshold | 0.80 |
| `--min-lines` | Minimum lines per window | 5 |
| `--k-confirm` | Consecutive positives for confirmation | 3 |

## Running Tests

```bash
make test
# or
pytest tests/ -v
```

## License

MIT
