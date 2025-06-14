# RVC ExecuTorch Experiment Notes

## UV Setup Guide

### Prerequisites

- Install [uv](https://docs.astral.sh/uv/) package manager
- A Python interpreter with version 3.9 or 3.10

### Setup Steps

```bash
# Install dependencies
uv sync

# Download models for the pipelines
uv run tools/download_models.py
```

### Run RVC

```bash
# Run the web interface
uv run infer-web.py

# Run desktop GUI
uv run gui_v1.py
```
