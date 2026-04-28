# AGENTS.md — Agentic AI Guidelines

> For manual setup, CLI commands, and general usage, see [README.md](README.md).

## Project Context

This is a CT kidney/tumor segmentation pipeline (KiTS19 dataset) using MONAI UNet on Intel XPU with Triton JIT. The agent should understand the codebase layout before making changes:

- `src/pipeline.py` — Operator classes (ingestion, pre-processing, inference, post-processing, validation, output)
- `src/infer.py` — CLI inference entry-point
- `src/visualize.py` — Montage & overlay rendering
- `src/benchmark.py` — Latency/throughput benchmarking
- `config.yaml` — Full pipeline configuration (cases, device, model paths, thresholds)
- `tests/run_all_unittest.py` — Full test suite (19 tests, synthetic data, no GPU required)

## Environment

- Python ≥ 3.10, dependencies in `requirements.txt`
- Intel XPU support is upstream in PyTorch ≥ 2.5 (no IPEX needed)
- KiTS19 data lives under `kits19/data/case_NNNNN/`
- Virtual environment: `.venv` (activate before running anything)

## Running Tests

After any code change, verify with:

```bash
python -m pytest tests/run_all_unittest.py -v
```

All 19 tests must pass. Tests use synthetic data and temp directories — no GPU required.

## Key Conventions

- **Device fallback**: Code must gracefully fall back to CPU when XPU is unavailable. Never hard-fail on missing XPU.
- **`torch.compile` fallback**: If `torch.compile` fails (e.g., unsupported backend), catch the exception and fall back to eager mode.
- **Output directories**: Predictions go to `output/predictions/`, reports to `output/reports/`, visualizations to `output/visualizations/`, benchmarks to `output/benchmarks/`.
- **Config-driven**: Case lists, device selection, model paths, and thresholds are all in `config.yaml`. Prefer config over hard-coded values.
- **NIfTI format**: Input volumes are `segmentation.nii.gz` (3D, shape `D×H×W`). Pre-processing: CT window `[-200, 300]`, normalise `(x-50)/250`.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: torch.xpu` | Upgrade to PyTorch ≥ 2.5 for native XPU support |
| `torch.xpu.is_available()` returns `False` | Check Intel GPU driver + oneAPI runtime |
| `FileNotFoundError` on NIfTI | Ensure `kits19/data/case_NNNNN/segmentation.nii.gz` exists |
| `torch.compile` fails on XPU | Falls back to eager mode automatically |
