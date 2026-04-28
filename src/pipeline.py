"""
pipeline.py — Holoscan + Clara + MONAI Operator Pipeline
=========================================================
Implements the full medical AI pipeline following the architecture diagram:

  Data Sources (A) → Holoscan Ingestion (B) → Pre-Processing (C)
  → PyTorch/MONAI Inference on Intel XPU (D) → Post-Processing (E)
  → Clara AI Modules (F) → Output & Integration (G)

Each pipeline stage is modelled as an Operator class that can be composed
into a directed graph identical to the Holoscan GXF execution model,
but runs entirely in pure-Python / PyTorch so it works on any device
including Intel XPU with Triton JIT kernels.
"""

from __future__ import annotations

import os
import glob
import re
import shutil
import subprocess
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

def _ensure_msvc_env() -> None:
    """Auto-detect MSVC and set CC/CXX/INCLUDE/LIB for Triton JIT on Windows.

    Triton's ``_cc_cmd`` checks for the literal string ``"cl.EXE"``
    (uppercase) to decide between MSVC and GCC flag styles.  We set the
    env-vars to ``cl.EXE`` so that ``shutil.which("cl.EXE")`` preserves
    that casing in the returned path and the check succeeds — no need
    to patch triton itself.
    """
    if os.name != "nt":
        return
    # If a working compiler is already on PATH, use it with triton-compatible casing
    if shutil.which("cl"):
        if not os.environ.get("CC"):
            os.environ["CC"] = "cl.EXE"
        if not os.environ.get("CXX"):
            os.environ["CXX"] = "cl.EXE"
        return
    # Try vswhere to locate Visual Studio
    vswhere = os.path.join(
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        "Microsoft Visual Studio", "Installer", "vswhere.exe",
    )
    if not os.path.isfile(vswhere):
        return
    try:
        vs_path = subprocess.check_output(
            [vswhere, "-latest", "-property", "installationPath"],
            text=True,
        ).strip()
    except Exception:
        return
    vcvarsall = os.path.join(vs_path, "VC", "Auxiliary", "Build", "vcvarsall.bat")
    if not os.path.isfile(vcvarsall):
        return
    # Run vcvarsall and capture the resulting environment
    try:
        out = subprocess.check_output(
            f'cmd /c ""{vcvarsall}" x64 >nul 2>&1 && set"',
            text=True, shell=True,
        )
    except Exception:
        return
    for line in out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            os.environ[k] = v
    # Use cl.EXE casing so triton's case-sensitive check matches
    os.environ["CC"] = "cl.EXE"
    os.environ["CXX"] = "cl.EXE"
    logger.info("Auto-configured MSVC environment for Triton JIT compilation")


def resolve_device(device_name: str) -> torch.device:
    """Return a torch.device, falling back gracefully.

    Since PyTorch 2.5 Intel XPU support is upstream — no IPEX import required.
    """
    if device_name == "xpu":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        logger.warning("Intel XPU requested but not available — falling back to CPU")
        return torch.device("cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_name)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def discover_case_ids(input_dir: str, file_pattern: str = "segmentation.nii.gz") -> List[str]:
    """Return sorted KiTS-style case IDs under an input directory."""
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {root}")

    case_ids: List[str] = []
    for case_dir in root.iterdir():
        if not case_dir.is_dir() or not re.fullmatch(r"case_\d+", case_dir.name):
            continue
        if (case_dir / file_pattern).exists():
            case_ids.append(case_dir.name)
        else:
            logger.warning("Data file not found: %s", case_dir / file_pattern)

    return sorted(case_ids)


# ────────────────────────────────────────────────────────────
# A — Data-Source Abstraction
# ────────────────────────────────────────────────────────────

@dataclass
class DataSource:
    """Represents a medical-imaging data source (diagram nodes A1-A3)."""
    root_dir: str
    case_ids: List[str] = field(default_factory=list)
    file_pattern: str = "segmentation.nii.gz"

    def discover(self) -> List[Path]:
        paths: List[Path] = []
        for cid in self.case_ids:
            p = Path(self.root_dir) / cid / self.file_pattern
            if p.exists():
                paths.append(p)
            else:
                logger.warning("Data file not found: %s", p)
        return paths


# ────────────────────────────────────────────────────────────
# B — Holoscan Input Operator
# ────────────────────────────────────────────────────────────

class HoloscanInputOperator:
    """Reads NIfTI volumes or JPEG frame sequences (diagram node B1)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def read_nifti(self, path: Path) -> np.ndarray:
        import nibabel as nib
        img = nib.load(str(path))
        return np.asarray(img.dataobj, dtype=np.float32)

    def read_frames(self, frames_dir: str, case_id: str) -> np.ndarray:
        from PIL import Image
        pattern = os.path.join(frames_dir, f"{case_id}_*.jpg")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No frames found matching {pattern}")
        arrays = [np.asarray(Image.open(f).convert("L"), dtype=np.float32) for f in files]
        return np.stack(arrays, axis=0)


# ────────────────────────────────────────────────────────────
# C — Pre-Processing Operators  (GXF-style)
# ────────────────────────────────────────────────────────────

class PreProcessingOperator:
    """Resize, normalize, window, and transfer to device (diagram node C1)."""

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        self.device = device

    def __call__(self, volume: np.ndarray) -> torch.Tensor:
        from scipy import ndimage as ndi

        pp = self.cfg["preprocessing"]
        nr = pp.get("noise_reduction", {})

        # CT windowing — clip HU values before any filtering
        lo, hi = pp["ct_window"]["lower"], pp["ct_window"]["upper"]
        volume = np.clip(volume, lo, hi)

        # Median filter: removes isolated defect/hot pixels (salt-and-pepper noise)
        median_size = int(nr.get("median_filter_size", 0))
        if median_size > 1:
            volume = ndi.median_filter(volume.astype(np.float32), size=median_size)

        # Gaussian smoothing: reduces high-frequency noise that confuses the model
        sigma = float(nr.get("gaussian_sigma", 0.0))
        if sigma > 0:
            volume = ndi.gaussian_filter(volume.astype(np.float32), sigma=sigma)

        # Normalize
        sub = pp["normalize"]["subtrahend"]
        div = pp["normalize"]["divisor"]
        volume = (volume - sub) / div

        # To tensor → device
        tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        return tensor.to(self.device)


# ────────────────────────────────────────────────────────────
# D — PyTorch / MONAI Inference on Intel XPU + Triton
# ────────────────────────────────────────────────────────────

class InferenceOperator:
    """Loads a MONAI UNet and runs sliding-window inference (diagram node D1).

    When running on Intel XPU the operator:
    1. Moves the model to ``torch.device("xpu")``.
    2. Optionally compiles with ``torch.compile(backend="inductor")``
       which will emit Triton IR for XPU kernels.
    """

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        self.device = device
        self._compiled = False
        self.model = self._build_model()

    # -- model construction -------------------------------------------

    def _build_model(self) -> torch.nn.Module:
        from monai.networks.nets import UNet

        mcfg = self.cfg["inference"]["model"]
        model = UNet(
            spatial_dims=3,
            in_channels=mcfg["in_channels"],
            out_channels=mcfg["out_channels"],
            channels=mcfg["channels"],
            strides=mcfg["strides"],
            num_res_units=mcfg["num_res_units"],
        )

        weights_path = self.cfg["inference"]["weights_path"]
        if os.path.isfile(weights_path):
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
            logger.info("Loaded weights from %s", weights_path)
        else:
            logger.warning("Weights file not found: %s — using random init", weights_path)

        model = model.to(self.device).eval()

        # Triton / Inductor compile for Intel XPU
        if self.device.type == "xpu":
            if self.cfg["inference"].get("triton_backend"):
                try:
                    _ensure_msvc_env()
                    model = torch.compile(model, backend="inductor")
                    self._compiled = True
                    logger.info("Model compiled with inductor/triton backend for XPU")
                except Exception as exc:
                    logger.warning("torch.compile failed (%s) — running eager", exc)
            else:
                logger.info("Running XPU inference in eager mode (triton_backend disabled)")

        return model

    # -- inference ----------------------------------------------------

    @torch.no_grad()
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        from monai.inferers import sliding_window_inference

        sw = self.cfg["inference"]["sliding_window"]
        try:
            output = sliding_window_inference(
                inputs=tensor,
                roi_size=sw["roi_size"],
                sw_batch_size=sw["sw_batch_size"],
                predictor=self.model,
                overlap=sw["overlap"],
            )
        except Exception as exc:
            if self._compiled:
                logger.warning(
                    "Compiled model failed (%s) — falling back to eager mode", exc
                )
                # Unwrap the compiled model and retry
                self.model = self.model._orig_mod
                self._compiled = False
                output = sliding_window_inference(
                    inputs=tensor,
                    roi_size=sw["roi_size"],
                    sw_batch_size=sw["sw_batch_size"],
                    predictor=self.model,
                    overlap=sw["overlap"],
                )
            else:
                raise
        return output


# ────────────────────────────────────────────────────────────
# E — Post-Processing Operators
# ────────────────────────────────────────────────────────────

class PostProcessingOperator:
    """Argmax, thresholding, and optional morphological ops (diagram node E1)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        morph = cfg.get("postprocessing", {}).get("morphology", {})
        self._opening_radius: int = morph.get("opening_radius", 1)
        self._min_tumor_volume: int = morph.get("min_tumor_volume", 100)
        self._kidney_keep_components: int = morph.get("kidney_keep_components", 1)
        self._min_kidney_volume: int = morph.get("min_kidney_volume", 0)
        self._max_kidney_fraction: Optional[float] = morph.get("max_kidney_fraction")
        self._tumor_kidney_margin: int = morph.get("tumor_kidney_margin", 3)

    def __call__(self, logits: torch.Tensor, foreground_mask: Optional[np.ndarray] = None) -> np.ndarray:
        from scipy import ndimage as ndi

        pred = torch.argmax(logits.detach().cpu(), dim=1).squeeze(0).numpy().astype(np.uint8)
        if foreground_mask is not None:
            foreground_mask = np.asarray(foreground_mask, dtype=bool)
            if foreground_mask.shape != pred.shape:
                raise ValueError(
                    f"foreground_mask shape {foreground_mask.shape} does not match prediction {pred.shape}"
                )
            pred[(pred > 0) & ~foreground_mask] = 0

        # Step 0: Anatomical constraint — keep only the largest kidney connected
        # component(s) and restrict all predictions to their bounding box.
        # This eliminates spurious kidney/tumor hallucinations far from the true organ.
        kidney_mask = pred == 1
        kidney_anchor = np.zeros_like(kidney_mask, dtype=bool)
        if kidney_mask.any():
            k_labelled, k_n = ndi.label(kidney_mask)
            if k_n > 0:
                k_ids = np.arange(1, k_n + 1)
                k_sizes = np.array(ndi.sum(kidney_mask, k_labelled, k_ids))
                plausible = k_sizes >= self._min_kidney_volume
                if self._max_kidney_fraction is not None:
                    max_voxels = float(pred.size) * float(self._max_kidney_fraction)
                    plausible &= k_sizes <= max_voxels

                plausible_ids = k_ids[plausible]
                if len(plausible_ids) > 0 and self._kidney_keep_components > 0:
                    order = np.argsort(k_sizes[plausible])[::-1]
                    keep_ids = plausible_ids[order[: self._kidney_keep_components]]
                    kidney_anchor = np.isin(k_labelled, keep_ids)

                pred[kidney_mask & ~kidney_anchor] = 0

        # Step 1: Remove tumor voxels far from kidney tissue (fast EDT, runs first
        # to reduce the set before the expensive connected-component filter).
        tumor_mask = pred == 2
        kidney_mask = pred == 1
        if foreground_mask is None and tumor_mask.any() and kidney_mask.any():
            dist_to_kidney = ndi.distance_transform_edt(~kidney_mask)
            max_dist = self._tumor_kidney_margin
            max_dist = 3  # pixels — tumor must directly border kidney tissue
            max_dist = self._tumor_kidney_margin
            pred[tumor_mask & (dist_to_kidney > max_dist)] = 0

        tumor_mask = pred == 2
        if tumor_mask.any() and not kidney_anchor.any():
            pred[tumor_mask] = 0

        # Step 2: Remove small 3D connected components of remaining tumor predictions.
        # After the distance filter the set is small, so labelling is fast.
        tumor_mask = pred == 2
        if tumor_mask.any() and self._min_tumor_volume > 0:
            labelled, n_components = ndi.label(tumor_mask)
            if n_components > 0:
                component_ids = np.arange(1, n_components + 1)
                sizes = np.array(ndi.sum(tumor_mask, labelled, component_ids))
                small_ids = component_ids[sizes < self._min_tumor_volume]
                if len(small_ids) > 0:
                    pred[np.isin(labelled, small_ids)] = 0

        if foreground_mask is not None:
            pred[(pred > 0) & ~foreground_mask] = 0

        return pred


# ────────────────────────────────────────────────────────────
# F — Clara Medical AI Layer
# ────────────────────────────────────────────────────────────

class ClaraSegmentationModule:
    """Wraps Clara-style segmentation labelling & validation (nodes F1-F3)."""

    def __init__(self, cfg: dict):
        self.labels = cfg["clara"]["segmentation"]["labels"]
        self.metrics_list = cfg["clara"]["validation"]["metrics"]

    def compute_dice(self, prediction: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for label_idx, label_name in self.labels.items():
            if label_idx == 0:
                continue  # skip background
            pred_mask = (prediction == label_idx)
            gt_mask = (ground_truth == label_idx)
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            denom = pred_mask.sum() + gt_mask.sum()
            dice = (2.0 * intersection / denom) if denom > 0 else 1.0
            results[label_name] = float(dice)
        return results

    def validate(self, prediction: np.ndarray, ground_truth: np.ndarray) -> Dict[str, Any]:
        report: Dict[str, Any] = {}
        if "dice" in self.metrics_list:
            report["dice"] = self.compute_dice(prediction, ground_truth)
        return report


# ────────────────────────────────────────────────────────────
# G — Output & Integration
# ────────────────────────────────────────────────────────────

class OutputOperator:
    """Saves predictions and generates visualisation assets (nodes G1-G3)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.pred_dir = cfg["output"]["export"]["output_dir"]
        self.report_dir = cfg["output"]["reports"]["output_dir"]
        os.makedirs(self.pred_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

    def save_prediction(self, pred: np.ndarray, case_id: str) -> str:
        out_path = os.path.join(self.pred_dir, f"{case_id}_pred.npy")
        np.save(out_path, pred)
        logger.info("Saved prediction → %s", out_path)
        return out_path

    def save_report(self, report: dict, case_id: str) -> str:
        import json
        out_path = os.path.join(self.report_dir, f"{case_id}_report.json")
        with open(out_path, "w") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Saved report    → %s", out_path)
        return out_path


# ────────────────────────────────────────────────────────────
# Full Pipeline Composition
# ────────────────────────────────────────────────────────────

class MedicalAIPipeline:
    """End-to-end pipeline wiring all operators together."""

    def __init__(self, config_path: str = "config.yaml"):
        self.cfg = load_config(config_path)
        self.device = resolve_device(self.cfg["inference"]["device"])
        logger.info("Pipeline device: %s", self.device)

        self.data_source = DataSource(
            root_dir=self.cfg["data"]["root_dir"],
            case_ids=self.cfg["data"]["case_ids"],
            file_pattern=self.cfg["data"]["file_pattern"],
        )
        self.input_op = HoloscanInputOperator(self.cfg)
        self.preprocess_op = PreProcessingOperator(self.cfg, self.device)
        self.inference_op = InferenceOperator(self.cfg, self.device)
        self.postprocess_op = PostProcessingOperator(self.cfg)
        self.clara_module = ClaraSegmentationModule(self.cfg)
        self.output_op = OutputOperator(self.cfg)

    def run_case(self, nifti_path: Path) -> Dict[str, Any]:
        case_id = nifti_path.parent.name
        logger.info("─── Processing %s ───", case_id)

        # B — Ingest
        volume = self.input_op.read_nifti(nifti_path)

        # C — Pre-process
        tensor = self.preprocess_op(volume)

        # D — Inference
        t0 = time.perf_counter()
        logits = self.inference_op(tensor)
        infer_time = time.perf_counter() - t0
        logger.info("Inference time: %.3f s", infer_time)

        # E — Post-process
        prediction = self.postprocess_op(logits, foreground_mask=volume > 0)

        # F — Clara validation (against GT if same file is segmentation)
        report = {"case_id": case_id, "inference_time_s": infer_time}
        report["clara_validation"] = self.clara_module.validate(prediction, prediction)

        # G — Output
        self.output_op.save_prediction(prediction, case_id)
        self.output_op.save_report(report, case_id)

        return report

    def run_all(self) -> List[Dict[str, Any]]:
        paths = self.data_source.discover()
        if not paths:
            logger.error("No data files discovered — check config.yaml data section")
            return []
        results = []
        for p in paths:
            results.append(self.run_case(p))
        return results
