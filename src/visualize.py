"""
visualize.py — Visualisation Utilities
========================================
Renders CT slices with segmentation overlays and generates
summary montages.  Corresponds to diagram node **G1**
(Visualization & UI — Surgical Overlay / Live Monitor / 3D Rendering).

Usage
-----
    python -m src.visualize                              # defaults
    python -m src.visualize --case case_00002 --slice 80
    python -m src.visualize --input kits19/data          # scan all case_* folders
    python -m src.visualize --prediction output/predictions/case_00002_pred.npy
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from src.pipeline import discover_case_ids, load_config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("visualize")


# ────────────────────────────────────────────────────────────
# Colour map for labels
# ────────────────────────────────────────────────────────────

DEFAULT_CMAP: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),        # background — transparent
    1: (255, 0, 0),      # kidney     — red
    2: (0, 255, 0),      # tumor      — green
}


KIDNEY_OUTLINE_COLOUR = np.array([0, 255, 255], dtype=np.uint8)
TUMOR_FILL_COLOUR = np.array([255, 0, 255], dtype=np.uint8)
TUMOR_BOUNDARY_COLOUR = np.array([255, 255, 0], dtype=np.uint8)


def label_to_rgb(mask: np.ndarray, cmap: Optional[Dict[int, Tuple[int, int, int]]] = None) -> np.ndarray:
    """Convert an integer label map to an RGB image (H, W, 3)."""
    if cmap is None:
        cmap = DEFAULT_CMAP
    h, w = mask.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label, colour in cmap.items():
        rgb[mask == label] = colour
    return rgb


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Return the inner outline of a 2-D binary mask."""
    from scipy import ndimage

    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return mask & ~eroded


def _slice_qc_warning(mask_slice: np.ndarray, tumor_kidney_margin: int = 3) -> Optional[str]:
    """Return a short warning when a slice has implausible segmentation geometry."""
    from scipy import ndimage

    total = mask_slice.size
    kidney = mask_slice == 1
    tumor = mask_slice == 2
    foreground_fraction = float((mask_slice > 0).sum()) / total
    kidney_fraction = float(kidney.sum()) / total

    if foreground_fraction > 0.45 or kidney_fraction > 0.35:
        return "QC: broad mask"
    if tumor.any() and not kidney.any():
        return "QC: tumor-only slice"
    if tumor.any():
        dist_to_kidney = ndimage.distance_transform_edt(~kidney)
        if float(dist_to_kidney[tumor].min()) > tumor_kidney_margin:
            return "QC: tumor away from kidney"
    return None


def constrain_tumor_to_foreground(prediction: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Return a display-safe prediction with labels only on nonzero source foreground."""
    constrained = prediction.copy()
    constrained[(constrained > 0) & ~(volume > 0)] = 0
    return constrained


def _is_label_like(slice_data: np.ndarray) -> bool:
    """Return True when the source slice looks like a discrete label mask."""
    unique = np.unique(slice_data)
    return len(unique) <= 8 and set(unique.astype(int).tolist()).issubset({0, 1, 2})


def _slice_title(slice_idx: int, mask_slice: np.ndarray, volume_slice: np.ndarray) -> str:
    """Build a title that makes tumor amount explicit."""
    pred_tumor = int((mask_slice == 2).sum())
    title = f"Slice {slice_idx} | tumor: {pred_tumor:,} px"
    if _is_label_like(volume_slice):
        gt_tumor = int((volume_slice == 2).sum())
        title = f"{title} (GT: {gt_tumor:,})"
    warning = _slice_qc_warning(mask_slice)
    if warning:
        title = f"{title} - {warning}"
    return title


def _figure_title(title: str, marker: bool = False) -> str:
    if marker:
        return f"{title}\nDebug markers: T shown only for connected tumor regions >= 10 px"
    return title


# ────────────────────────────────────────────────────────────
# Tumor annotation
# ────────────────────────────────────────────────────────────

def _annotate_tumors(ax: plt.Axes, mask_slice: np.ndarray) -> None:
    """Add compact tumor markers at connected tumor-region centroids."""
    from scipy import ndimage

    tumor_mask = (mask_slice == 2).astype(np.uint8)
    if tumor_mask.sum() == 0:
        return

    labelled, n_regions = ndimage.label(tumor_mask)
    for region_id in range(1, n_regions + 1):
        ys, xs = np.where(labelled == region_id)
        if len(ys) < 10:
            continue
        cy, cx = float(ys.mean()), float(xs.mean())
        ax.text(
            cx,
            cy,
            "T",
            fontsize=8,
            fontweight="bold",
            color="yellow",
            ha="center",
            va="center",
            bbox=dict(boxstyle="square,pad=0.1", facecolor="black", edgecolor="yellow", linewidth=0.4),
        )


# ────────────────────────────────────────────────────────────
# Single-slice overlay
# ────────────────────────────────────────────────────────────

def overlay_slice(
    ct_slice: np.ndarray,
    mask_slice: np.ndarray,
    alpha: float = 0.4,
    cmap: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """Blend a CT grey-scale slice with an RGB label overlay."""
    # Normalise CT to 0-255
    mn, mx = ct_slice.min(), ct_slice.max()
    if mx - mn > 0:
        ct_norm = ((ct_slice - mn) / (mx - mn) * 255).astype(np.uint8)
    else:
        ct_norm = np.zeros_like(ct_slice, dtype=np.uint8)
    ct_rgb = np.stack([ct_norm] * 3, axis=-1)

    mask_rgb = label_to_rgb(mask_slice, cmap)
    # Fill tumor and other non-kidney labels; kidney is drawn as an outline so
    # broad false-positive kidney masks remain easy to see.
    fg = (mask_slice > 0) & (mask_slice != 1)
    blended = ct_rgb.copy()
    blended[fg] = (
        (1 - alpha) * ct_rgb[fg].astype(np.float32)
        + alpha * mask_rgb[fg].astype(np.float32)
    ).astype(np.uint8)
    blended[_mask_boundary(mask_slice == 1)] = KIDNEY_OUTLINE_COLOUR
    tumor = mask_slice == 2
    blended[tumor] = TUMOR_FILL_COLOUR
    blended[_mask_boundary(tumor)] = TUMOR_BOUNDARY_COLOUR
    return blended


# ────────────────────────────────────────────────────────────
# Montage of axial slices
# ────────────────────────────────────────────────────────────

def render_montage(
    volume: np.ndarray,
    prediction: np.ndarray,
    case_id: str,
    output_dir: str,
    n_slices: int = 9,
    alpha: float = 0.4,
    marker: bool = False,
) -> str:
    """Save an NxN grid of axial slices with overlay."""
    os.makedirs(output_dir, exist_ok=True)
    prediction = constrain_tumor_to_foreground(prediction, volume)
    depth = volume.shape[0]
    indices = np.linspace(0, depth - 1, n_slices, dtype=int)

    cols = int(np.ceil(np.sqrt(n_slices)))
    rows = int(np.ceil(n_slices / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_2d(axes)
    warnings: List[str] = []

    for idx, ax in enumerate(axes.flat):
        if idx < n_slices:
            si = indices[idx]
            blended = overlay_slice(volume[si], prediction[si], alpha=alpha)
            ax.imshow(blended)
            if marker:
                _annotate_tumors(ax, prediction[si])
            title = _slice_title(si, prediction[si], volume[si])
            warning = _slice_qc_warning(prediction[si])
            if warning:
                warnings.append(warning)
                ax.set_title(title, fontsize=8, color="darkorange")
            else:
                ax.set_title(title, fontsize=8)
        ax.axis("off")

    fig.suptitle(f"{case_id} — Segmentation Overlay", fontsize=13)
    fig.suptitle(_figure_title(f"{case_id} - Segmentation Overlay", marker=marker), fontsize=13)
    handles = [
        Patch(facecolor=KIDNEY_OUTLINE_COLOUR / 255.0, label="kidney outline"),
        Patch(facecolor=TUMOR_FILL_COLOUR / 255.0, label="tumor"),
        Patch(facecolor=TUMOR_BOUNDARY_COLOUR / 255.0, label="tumor edge"),
    ]
    axes.flat[0].legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.7)

    fig.tight_layout()
    out_path = os.path.join(output_dir, f"{case_id}_montage.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved montage → %s", out_path)
    return out_path


# ────────────────────────────────────────────────────────────
# Side-by-side comparison
# ────────────────────────────────────────────────────────────

def render_comparison(
    volume: np.ndarray,
    prediction: np.ndarray,
    slice_idx: int,
    case_id: str,
    output_dir: str,
    alpha: float = 0.4,
    marker: bool = False,
) -> str:
    """Save a side-by-side image: CT | Overlay | Label mask."""
    os.makedirs(output_dir, exist_ok=True)
    prediction = constrain_tumor_to_foreground(prediction, volume)
    ct_slice = volume[slice_idx]
    mask_slice = prediction[slice_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # CT
    axes[0].imshow(ct_slice, cmap="gray")
    axes[0].set_title("CT")

    # Overlay
    blended = overlay_slice(ct_slice, mask_slice, alpha)
    axes[1].imshow(blended)
    if marker:
        _annotate_tumors(axes[1], mask_slice)
    axes[1].set_title(_slice_title(slice_idx, mask_slice, ct_slice))

    # Label mask
    axes[2].imshow(overlay_slice((ct_slice > 0).astype(np.float32), mask_slice, alpha=1.0))
    if marker:
        _annotate_tumors(axes[2], mask_slice)
    axes[2].set_title("Segmentation Mask")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"{case_id} — Slice {slice_idx}", fontsize=13)
    fig.suptitle(_figure_title(f"{case_id} - Slice {slice_idx}", marker=marker), fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"{case_id}_slice{slice_idx}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved comparison → %s", out_path)
    return out_path


# ────────────────────────────────────────────────────────────
# Visualise from saved prediction + original volume
# ────────────────────────────────────────────────────────────

def visualize_case(
    config_path: str = "config.yaml",
    case_id: str = "case_00002",
    prediction_path: Optional[str] = None,
    slice_idx: Optional[int] = None,
    marker: bool = False,
    input_dir: Optional[str] = None,
) -> List[str]:
    """High-level helper: load volume + prediction, produce images."""
    cfg = load_config(config_path)
    root = input_dir or cfg["data"]["root_dir"]
    pattern = cfg["data"]["file_pattern"]
    output_dir = os.path.join(cfg["output"]["visualization"]["output_dir"], case_id)
    alpha = cfg["output"]["visualization"]["overlay_alpha"]

    # Load original volume
    nifti_path = Path(root) / case_id / pattern
    if not nifti_path.exists():
        raise FileNotFoundError(f"Volume not found: {nifti_path}")

    import nibabel as nib
    volume = np.asarray(nib.load(str(nifti_path)).dataobj, dtype=np.float32)

    # Load prediction
    if prediction_path is None:
        prediction_path = os.path.join(
            cfg["output"]["export"]["output_dir"], f"{case_id}_pred.npy"
        )
    if not os.path.exists(prediction_path):
        raise FileNotFoundError(f"Prediction not found: {prediction_path}")

    prediction = np.load(prediction_path)
    logger.info("Volume shape: %s  Prediction shape: %s", volume.shape, prediction.shape)

    outputs: List[str] = []

    # Montage
    outputs.append(render_montage(volume, prediction, case_id, output_dir, alpha=alpha, marker=marker))

    # Single-slice comparison
    if slice_idx is None:
        # pick the slice with the most non-zero predictions
        nonzero_per_slice = (prediction > 0).reshape(prediction.shape[0], -1).sum(axis=1)
        slice_idx = int(np.argmax(nonzero_per_slice))
    outputs.append(render_comparison(volume, prediction, slice_idx, case_id, output_dir, alpha=alpha, marker=marker))

    return outputs


def visualize_cases(
    config_path: str = "config.yaml",
    input_dir: Optional[str] = None,
    case_id: Optional[str] = None,
    slice_idx: Optional[int] = None,
    marker: bool = False,
) -> List[str]:
    """Visualize one case, configured cases, or all case_* folders in input_dir."""
    cfg = load_config(config_path)
    if input_dir:
        cases = (
            [case_id]
            if case_id
            else discover_case_ids(input_dir, cfg["data"]["file_pattern"])
        )
    else:
        cases = [case_id] if case_id else cfg["data"]["case_ids"]

    outputs: List[str] = []
    for cid in cases:
        try:
            outputs.extend(
                visualize_case(
                    config_path=config_path,
                    case_id=cid,
                    prediction_path=None,
                    slice_idx=slice_idx,
                    marker=marker,
                    input_dir=input_dir,
                )
            )
        except FileNotFoundError as exc:
            logger.warning("Skipping %s - %s", cid, exc)
    return outputs


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise segmentation results")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--case", default=None)
    parser.add_argument("--input", default=None, help="Folder containing case_* subfolders")
    parser.add_argument("--prediction", default=None, help="Path to .npy prediction")
    parser.add_argument("--slice", type=int, default=None, help="Axial slice index")
    parser.add_argument("--marker", dest="marker", action="store_true", help="Show debug tumor markers")
    args = parser.parse_args()

    if args.input and args.prediction:
        parser.error("--prediction can only be used with a single --case, not --input")

    if args.input:
        paths = visualize_cases(
            config_path=args.config,
            input_dir=args.input,
            case_id=args.case,
            slice_idx=args.slice,
            marker=args.marker,
        )
    else:
        paths = visualize_case(
            config_path=args.config,
            case_id=args.case or "case_00002",
            prediction_path=args.prediction,
            slice_idx=args.slice,
            marker=args.marker,
        )
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
