# Medical AI Pipeline — CT Kidney Tumor Segmentation

> **Holoscan-style PyTorch + Clara-style Medical AI Pipeline on Intel XPU / Triton**

End-to-end inference pipeline for 3D CT kidney/tumor segmentation on the
[KiTS19](https://github.com/neheller/kits19) dataset, using
**MONAI**, **Clara-style** validation concepts, and a
**Holoscan-style** staged operator-pipeline architecture — all accelerated on
**Intel XPU** (native since PyTorch 2.5) and **Triton JIT kernels**
(`torch.compile(backend="inductor")`).

This repository does not import or execute NVIDIA Holoscan or NVIDIA Clara
runtime APIs. Its ingestion, pre-processing, and validation stages use standard
Python libraries (`nibabel`, `numpy`, `scipy`) and PyTorch tensors, while
keeping the code organized in operator-style stages that resemble a
Holoscan/GXF graph and Clara-style metric validation.

---

## Architecture

The pipeline follows a Holoscan-style + Clara-style reference architecture, but
runs as a pure Python/PyTorch batch inference workflow:

```
Data Sources (A)          NIfTI volumes / JPEG frames / DICOM
        │
Input Reading (B)         nibabel NIfTI reader / PIL frame reader
        │
Pre-Processing (C)        CT windowing, normalisation, GPU memory transfer
        │
AI Inference (D)          MONAI UNet — Intel XPU + Triton JIT
        │
Post-Processing (E)       Argmax, thresholding, morphology
        │
Clara-style Validation (F) Segmentation labelling, Dice validation
        │
Output & Integration (G)  Visualisation overlays, NIfTI export, JSON reports
```

## Why Holoscan Is Not Used at Runtime

KiTS19 is a stored dataset on disk, so this repo performs offline/batch
inference over NIfTI files rather than operating as a real-time deployed
medical application. For that workflow, Holoscan cannot be meaningfully
utilized without adding a live data source, a Holoscan application graph, and
runtime operators around the existing Python/PyTorch processing.

Holoscan would be useful if this were a deployed application with things like:

- real-time imaging streams
- video/frame grabbers
- sensor input
- GPU memory message passing between operators
- a production operator graph/runtime
- integration with medical-device I/O

## Clara-Style Validation

The validation stage is also Clara-style rather than Clara-backed. In this repo,
`ClaraSegmentationModule` computes segmentation labels and Dice metrics locally
with NumPy. It does not call NVIDIA Clara runtime services, model lifecycle
management, deployment tooling, or Clara application APIs.

## Project Structure

```
monai/
├── config.yaml                  # Full pipeline configuration
├── requirements.txt             # Python dependencies
├── unet_kits19_state_dict.pth   # Downloaded locally; not included in Git
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
├── kits19/                      # KiTS19 dataset cloned locally by developers
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

The KiTS19 dataset is not included in this repository. Developers must clone
the dataset themselves from [neheller/kits19](https://github.com/neheller/kits19)
and place it under `kits19/` so case data is available at
`kits19/data/case_NNNNN/segmentation.nii.gz`.

## Pre-processing Pipeline

1. **CT Windowing** — clip HU values to `[-200, 300]`
2. **Normalisation** — `(x - 50) / 250`
3. **Tensor conversion** — `numpy → torch.Tensor` with batch/channel dims
4. **Device transfer** — move to the resolved device (`auto`, CPU, CUDA, XPU, or MPS)

## Model Architecture

**MONAI 3D UNet** with residual units:

- `in_channels`: 1 (single CT channel)
- `out_channels`: 3 (background + kidney + tumor)
- `channels`: [16, 32, 64, 128, 256]
- `strides`: [2, 2, 2, 2]
- `num_res_units`: 2

Pre-trained weights are not included in this repository. Download
`unet_kits19_state_dict.pth` from Intel's public OpenVINO model server:

<https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/kidney-segmentation-kits19/>

Place the file at the repository root, or update `model.path` in
`config.yaml` to point to your local copy. For more context on loading this
PyTorch model, see the OpenVINO notebook section:

<https://docs.openvino.ai/2024/notebooks/ct-segmentation-quantize-nncf-with-output.html#load-pytorch-model>

## Device Selection + Triton Acceleration

When `inference.triton_backend: true` in `config.yaml`:

1. `inference.device: "auto"` selects the first available accelerator, then CPU
2. Explicit `cuda`, `xpu`, `mps`, or `cpu` values are validated before use
3. `torch.compile(backend="inductor")` is attempted for Intel XPU and falls back to eager mode if compilation fails

Requires PyTorch >= 2.5 (XPU support is upstream -- no IPEX needed)


## Inference Flow

```bash
# Run on the best available device (default from config.yaml)
python -m src.infer

# Specific case on CPU
python -m src.infer --case case_00002 --device cpu

# Batch inference over every case_* folder in kits19/data
python -m src.infer --input kits19/data --device xpu
```

Batch inference scans `kits19/data` for `case_*` folders, runs the configured
MONAI UNet pipeline for each case, and writes predictions to
`output/predictions/` with filenames such as `case_00002_pred.npy`. Reports are
written to `output/reports/`.

## Evaluation Metric — Dice Score

Per-class Sørensen–Dice coefficient:

$$\text{Dice}(P, G) = \frac{2 |P \cap G|}{|P| + |G|}$$

Computed by `ClaraSegmentationModule.compute_dice()` in `src/pipeline.py`.

## Visualisation

```bash
# Generate montage + comparison images
python -m src.visualize --case case_00002

# Visualize one case at a specific axial slice
python -m src.visualize --case case_00002 --slice 80

# Batch visualization over every case_* folder in kits19/data
python -m src.visualize --input kits19/data
```

Batch visualization pairs each case with its corresponding prediction from
`output/predictions/` and writes PNG montage/comparison images to
`output/visualizations/`.

## Benchmarking

```bash
# Single device
python -m src.benchmark --device auto --warmup 3 --runs 10

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
| **Acceleration** | CUDA, Intel XPU, or MPS when available; optional XPU Triton via `torch.compile(backend="inductor")` |
| **Fallback** | Graceful degradation to CPU if the requested backend is unavailable |
| **Model format** | Native PyTorch `state_dict` (no TensorRT dependency) |
| **DICOM export** | Extend `OutputOperator` to write DICOM SEG / SR |
| **Containerisation** | Standard Docker with `intel/oneapi-basekit` base image |
| **Scaling** | Current repo is single-process Python/PyTorch; a deployed Holoscan app could add an operator DAG for multi-GPU / multi-node workflows |
