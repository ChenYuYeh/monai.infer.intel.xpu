# Medical AI Pipeline — CT Kidney Tumor Segmentation

> **Holoscan + PyTorch + Clara Medical AI Pipeline on Intel XPU / Triton**

End-to-end inference pipeline for 3D CT kidney/tumor segmentation on the
[KiTS19](https://github.com/neheller/kits19) dataset, using
**MONAI**, **NVIDIA Clara** validation concepts, and the
**Holoscan** operator-pipeline architecture — all accelerated on
**Intel XPU** (native since PyTorch 2.5) and **Triton JIT kernels**
(`torch.compile(backend="inductor")`).

---

## Architecture

The pipeline follows the NVIDIA Holoscan + Clara reference architecture:

```
Data Sources (A)          NIfTI volumes / JPEG frames / DICOM
        │
Holoscan Ingestion (B)    Frame Grabber, NIfTI Reader, Sensor Adapter
        │
Pre-Processing (C)        CT windowing, normalisation, GPU memory transfer
        │
AI Inference (D)          MONAI UNet — Intel XPU + Triton JIT
        │
Post-Processing (E)       Argmax, thresholding, morphology
        │
Clara AI Layer (F)        Segmentation labelling, Dice validation, lifecycle
        │
Output & Integration (G)  Visualisation overlays, NIfTI export, JSON reports
```

## Project Structure

```
monai/
├── config.yaml                  # Full pipeline configuration
├── requirements.txt             # Python dependencies
├── unet_kits19_state_dict.pth   # Pre-trained MONAI UNet weights
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py              # Operator classes (B–G) + pipeline composition
│   ├── infer.py                 # CLI inference entry-point
│   ├── visualize.py             # Montage & overlay rendering
│   └── benchmark.py             # Latency/throughput benchmarking
│
├── tests/
│   ├── __init__.py
│   └── run_all_unittest.py      # Full test-suite
│
├── kits19/                      # KiTS19 dataset (NIfTI volumes)
│   └── data/case_NNNNN/segmentation.nii.gz
│
├── output/                      # Generated at runtime
│   ├── predictions/
│   ├── visualizations/
│   ├── reports/
│   └── benchmarks/
│
├── README.md
├── AGENTS.md
└── medical_infrastructure.md    # Architecture diagram (Mermaid)
```

## Medical Imaging Task

**Kidney tumor segmentation** on contrast-enhanced CT scans from KiTS19.
The model outputs per-voxel labels:

| Label | Class       |
|-------|-------------|
| 0     | Background  |
| 1     | Kidney      |
| 2     | Tumor       |

## Dataset Format

KiTS19 provides 3D NIfTI volumes (`segmentation.nii.gz`) per case.
Each volume is a stack of axial slices with shape `(D, H, W)`.

## Pre-processing Pipeline

1. **CT Windowing** — clip HU values to `[-200, 300]`
2. **Normalisation** — `(x - 50) / 250`
3. **Tensor conversion** — `numpy → torch.Tensor` with batch/channel dims
4. **Device transfer** — move to Intel XPU (or CPU/CUDA fallback)

## Model Architecture

**MONAI 3D UNet** with residual units:

- `in_channels`: 1 (single CT channel)
- `out_channels`: 3 (background + kidney + tumor)
- `channels`: [16, 32, 64, 128, 256]
- `strides`: [2, 2, 2, 2]
- `num_res_units`: 2

Pre-trained weights: `unet_kits19_state_dict.pth`

## Intel XPU + Triton Acceleration

When `inference.triton_backend: true` in `config.yaml`:

1. The model is moved to `torch.device("xpu")`
2. `torch.compile(backend="inductor")` generates **Triton IR** kernels
   optimised for Intel GPU (Arc / Data Center Max / Flex)
3. Fallback to CPU if XPU is not available

Requires PyTorch >= 2.5 (XPU support is upstream -- no IPEX needed)


## Inference Flow

```bash
# Run on Intel XPU (default from config.yaml)
python -m src.infer

# Specific case on CPU
python -m src.infer --case case_00002 --device cpu
```

## Evaluation Metric — Dice Score

Per-class Sørensen–Dice coefficient:

$$\text{Dice}(P, G) = \frac{2 |P \cap G|}{|P| + |G|}$$

Computed by `ClaraSegmentationModule.compute_dice()` in `src/pipeline.py`.

## Visualisation

```bash
# Generate montage + comparison images
python -m src.visualize --case case_00002
```

Outputs PNG files to `output/visualizations/`.

## Benchmarking

```bash
# Single device
python -m src.benchmark --device xpu --warmup 3 --runs 10

# Compare all available devices
python -m src.benchmark --compare
```

Reports latency percentiles (p50/p90/p95/p99), throughput (FPS), and memory.

## Running Tests

```bash
python -m pytest tests/run_all_unittest.py -v
```

## Deployment Considerations

| Concern | Approach |
|---------|----------|
| **Acceleration** | Intel XPU + Triton via `torch.compile(backend="inductor")` |
| **Fallback** | Graceful degradation to CPU if XPU unavailable |
| **Model format** | Native PyTorch `state_dict` (no TensorRT dependency) |
| **DICOM export** | Extend `OutputOperator` to write DICOM SEG / SR |
| **Containerisation** | Standard Docker with `intel/oneapi-basekit` base image |
| **Scaling** | Holoscan operator DAG supports multi-GPU / multi-node |
