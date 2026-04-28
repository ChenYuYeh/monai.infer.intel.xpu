"""
unittest/run_all_unittest.py — Test-suite for the Medical AI Pipeline
======================================================================
Covers every pipeline operator (B–G) and the public APIs in
``src.infer``, ``src.visualize``, and ``src.benchmark``.

Run
---
    python -m pytest unittest/run_all_unittest.py -v
    python -m unittest unittest.run_all_unittest -v
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import yaml


# ────────────────────────────────────────────────────────────
# Helpers — minimal config for testing
# ────────────────────────────────────────────────────────────

def _make_test_config(tmpdir: str) -> dict:
    """Return a config dict pointing at *tmpdir* for all I/O."""
    return {
        "pipeline": {"name": "test", "version": "0.0.1", "description": "unit test"},
        "data": {
            "root_dir": tmpdir,
            "case_ids": ["case_00000"],
            "file_pattern": "segmentation.nii.gz",
        },
        "input_operators": {
            "nifti_reader": {"enabled": True, "file_extension": ".nii.gz"},
            "frame_reader": {"enabled": True, "file_extension": ".jpg", "batch_size": 4},
        },
        "preprocessing": {
            "spatial_size": [32, 32, 16],
            "ct_window": {"lower": -200.0, "upper": 300.0},
            "normalize": {"subtrahend": 50.0, "divisor": 250.0},
            "spacing": [1.0, 1.0, 1.0],
            "orientation": "RAS",
        },
        "inference": {
            "device": "cpu",
            "triton_backend": False,
            "model": {
                "architecture": "UNet",
                "in_channels": 1,
                "out_channels": 3,
                "channels": [8, 16],
                "strides": [2],
                "num_res_units": 0,
            },
            "weights_path": os.path.join(tmpdir, "dummy_weights.pth"),
            "sliding_window": {
                "roi_size": [32, 32, 16],
                "sw_batch_size": 1,
                "overlap": 0.25,
            },
        },
        "postprocessing": {
            "argmax_dim": 0,
            "threshold": 0.5,
            "morphology": {"closing_radius": 2, "opening_radius": 1},
            "temporal_smoothing": {"enabled": False, "window_size": 3},
        },
        "clara": {
            "segmentation": {
                "labels": {0: "background", 1: "kidney", 2: "tumor"},
            },
            "validation": {
                "metrics": ["dice"],
                "confidence_threshold": 0.85,
            },
            "deployment": {
                "model_format": "pytorch",
                "export_onnx": False,
            },
        },
        "output": {
            "visualization": {
                "enabled": True,
                "output_dir": os.path.join(tmpdir, "vis"),
                "overlay_alpha": 0.4,
                "colormap": {"kidney": [255, 0, 0], "tumor": [0, 255, 0]},
            },
            "export": {"output_dir": os.path.join(tmpdir, "pred")},
            "reports": {
                "output_dir": os.path.join(tmpdir, "reports"),
                "save_metrics_csv": True,
            },
        },
    }


def _write_config(tmpdir: str, cfg: dict) -> str:
    path = os.path.join(tmpdir, "config.yaml")
    with open(path, "w") as fh:
        yaml.dump(cfg, fh)
    return path


# ────────────────────────────────────────────────────────────
# Tests — Pipeline Operators
# ────────────────────────────────────────────────────────────

class TestResolveDevice(unittest.TestCase):
    def test_cpu(self):
        from src.pipeline import resolve_device
        dev = resolve_device("cpu")
        self.assertEqual(dev.type, "cpu")

    def test_unknown_xpu_falls_back(self):
        from src.pipeline import resolve_device
        dev = resolve_device("xpu")
        # Should be xpu if available, else cpu
        self.assertIn(dev.type, ("xpu", "cpu"))


class TestPreProcessingOperator(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cfg = _make_test_config(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_output_shape_and_device(self):
        from src.pipeline import PreProcessingOperator
        op = PreProcessingOperator(self.cfg, torch.device("cpu"))
        vol = np.random.randn(16, 32, 32).astype(np.float32)
        tensor = op(vol)
        self.assertEqual(tensor.shape, (1, 1, 16, 32, 32))
        self.assertEqual(tensor.device.type, "cpu")

    def test_ct_windowing(self):
        from src.pipeline import PreProcessingOperator
        op = PreProcessingOperator(self.cfg, torch.device("cpu"))
        vol = np.array([[-500.0, 0.0, 500.0]], dtype=np.float32).reshape(1, 1, 3)
        tensor = op(vol)
        # After clip to [-200, 300] → [-200, 0, 300]
        # After normalize: (-200-50)/250, (0-50)/250, (300-50)/250
        expected_min = (-200 - 50) / 250
        expected_max = (300 - 50) / 250
        self.assertAlmostEqual(tensor.min().item(), expected_min, places=4)
        self.assertAlmostEqual(tensor.max().item(), expected_max, places=4)


class TestInferenceOperator(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cfg = _make_test_config(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_forward_shape(self):
        from src.pipeline import InferenceOperator
        op = InferenceOperator(self.cfg, torch.device("cpu"))
        x = torch.randn(1, 1, 32, 32, 16)
        out = op(x)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 3)  # out_channels

    def test_model_is_eval(self):
        from src.pipeline import InferenceOperator
        op = InferenceOperator(self.cfg, torch.device("cpu"))
        self.assertFalse(op.model.training)


class TestPostProcessingOperator(unittest.TestCase):
    def _logits_from_labels(self, labels: np.ndarray) -> torch.Tensor:
        logits = torch.zeros(1, 3, *labels.shape)
        for label in range(3):
            logits[0, label][torch.from_numpy(labels == label)] = 5.0
        return logits

    def test_argmax(self):
        from src.pipeline import PostProcessingOperator
        cfg = _make_test_config(tempfile.mkdtemp())
        op = PostProcessingOperator(cfg)
        logits = torch.zeros(1, 3, 4, 4, 4)
        logits[0, 1, :, :, :] = 1.0  # class 1 dominant
        pred = op(logits)
        self.assertTrue((pred == 1).all())
        shutil.rmtree(cfg["data"]["root_dir"])

    def test_far_tumor_removed(self):
        from src.pipeline import PostProcessingOperator
        cfg = _make_test_config(tempfile.mkdtemp())
        cfg["postprocessing"]["morphology"].update({
            "min_tumor_volume": 0,
            "kidney_keep_components": 2,
            "tumor_kidney_margin": 1,
        })
        labels = np.zeros((12, 12, 12), dtype=np.uint8)
        labels[2:5, 2:5, 2:5] = 1
        labels[9:11, 9:11, 9:11] = 2
        pred = PostProcessingOperator(cfg)(self._logits_from_labels(labels))
        self.assertEqual(int((pred == 2).sum()), 0)
        shutil.rmtree(cfg["data"]["root_dir"])

    def test_adjacent_tumor_preserved(self):
        from src.pipeline import PostProcessingOperator
        cfg = _make_test_config(tempfile.mkdtemp())
        cfg["postprocessing"]["morphology"].update({
            "min_tumor_volume": 0,
            "kidney_keep_components": 2,
            "tumor_kidney_margin": 1,
        })
        labels = np.zeros((12, 12, 12), dtype=np.uint8)
        labels[2:5, 2:5, 2:5] = 1
        labels[5:7, 3:5, 3:5] = 2
        pred = PostProcessingOperator(cfg)(self._logits_from_labels(labels))
        self.assertGreater(int((pred == 2).sum()), 0)
        shutil.rmtree(cfg["data"]["root_dir"])

    def test_foreground_mask_removes_background_tumor(self):
        from src.pipeline import PostProcessingOperator
        cfg = _make_test_config(tempfile.mkdtemp())
        cfg["postprocessing"]["morphology"].update({
            "min_tumor_volume": 0,
            "kidney_keep_components": 2,
            "tumor_kidney_margin": 1,
        })
        labels = np.zeros((12, 12, 12), dtype=np.uint8)
        labels[2:5, 2:5, 2:5] = 1
        labels[5:7, 3:5, 3:5] = 2
        foreground = labels == 1
        pred = PostProcessingOperator(cfg)(self._logits_from_labels(labels), foreground_mask=foreground)
        self.assertEqual(int((pred == 2).sum()), 0)
        shutil.rmtree(cfg["data"]["root_dir"])

    def test_foreground_mask_preserves_foreground_tumor_without_kidney_contact(self):
        from src.pipeline import PostProcessingOperator
        cfg = _make_test_config(tempfile.mkdtemp())
        cfg["postprocessing"]["morphology"].update({
            "min_tumor_volume": 0,
            "kidney_keep_components": 2,
            "tumor_kidney_margin": 1,
        })
        labels = np.zeros((12, 12, 12), dtype=np.uint8)
        labels[2:4, 2:4, 2:4] = 1
        labels[8:10, 8:10, 8:10] = 2
        foreground = labels > 0
        pred = PostProcessingOperator(cfg)(self._logits_from_labels(labels), foreground_mask=foreground)
        self.assertEqual(int((pred == 2).sum()), 8)
        shutil.rmtree(cfg["data"]["root_dir"])

    def test_overlarge_kidney_rejected(self):
        from src.pipeline import PostProcessingOperator
        cfg = _make_test_config(tempfile.mkdtemp())
        cfg["postprocessing"]["morphology"].update({
            "max_kidney_fraction": 0.2,
            "min_tumor_volume": 0,
        })
        labels = np.zeros((10, 10, 10), dtype=np.uint8)
        labels[:8, :, :] = 1
        labels[8:10, 8:10, 8:10] = 2
        pred = PostProcessingOperator(cfg)(self._logits_from_labels(labels))
        self.assertEqual(int((pred == 1).sum()), 0)
        self.assertEqual(int((pred == 2).sum()), 0)
        shutil.rmtree(cfg["data"]["root_dir"])

    def test_two_kidney_components_retained(self):
        from src.pipeline import PostProcessingOperator
        cfg = _make_test_config(tempfile.mkdtemp())
        cfg["postprocessing"]["morphology"].update({
            "kidney_keep_components": 2,
            "min_tumor_volume": 0,
        })
        labels = np.zeros((12, 12, 12), dtype=np.uint8)
        labels[2:4, 2:4, 2:4] = 1
        labels[8:10, 8:10, 8:10] = 1
        pred = PostProcessingOperator(cfg)(self._logits_from_labels(labels))
        self.assertEqual(int((pred == 1).sum()), 16)
        shutil.rmtree(cfg["data"]["root_dir"])


class TestClaraSegmentationModule(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cfg = _make_test_config(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_dice_perfect(self):
        from src.pipeline import ClaraSegmentationModule
        mod = ClaraSegmentationModule(self.cfg)
        mask = np.array([0, 1, 1, 2, 2, 0], dtype=np.uint8)
        result = mod.compute_dice(mask, mask)
        self.assertAlmostEqual(result["kidney"], 1.0)
        self.assertAlmostEqual(result["tumor"], 1.0)

    def test_dice_zero(self):
        from src.pipeline import ClaraSegmentationModule
        mod = ClaraSegmentationModule(self.cfg)
        pred = np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8)
        gt = np.array([0, 0, 0, 2, 2, 0], dtype=np.uint8)
        result = mod.compute_dice(pred, gt)
        self.assertAlmostEqual(result["kidney"], 0.0)
        self.assertAlmostEqual(result["tumor"], 0.0)

    def test_validate_returns_dice(self):
        from src.pipeline import ClaraSegmentationModule
        mod = ClaraSegmentationModule(self.cfg)
        mask = np.ones(10, dtype=np.uint8)
        report = mod.validate(mask, mask)
        self.assertIn("dice", report)


class TestOutputOperator(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cfg = _make_test_config(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_save_prediction(self):
        from src.pipeline import OutputOperator
        op = OutputOperator(self.cfg)
        pred = np.zeros((4, 4, 4), dtype=np.uint8)
        path = op.save_prediction(pred, "case_test")
        self.assertTrue(os.path.isfile(path))

    def test_save_report(self):
        from src.pipeline import OutputOperator
        op = OutputOperator(self.cfg)
        path = op.save_report({"dice": 0.95}, "case_test")
        self.assertTrue(os.path.isfile(path))
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["dice"], 0.95)


# ────────────────────────────────────────────────────────────
# Tests — Visualize
# ────────────────────────────────────────────────────────────

class TestVisualize(unittest.TestCase):
    def test_label_to_rgb(self):
        from src.visualize import label_to_rgb
        mask = np.array([[0, 1], [2, 0]], dtype=np.uint8)
        rgb = label_to_rgb(mask)
        self.assertEqual(rgb.shape, (2, 2, 3))
        np.testing.assert_array_equal(rgb[0, 1], [255, 0, 0])  # kidney
        np.testing.assert_array_equal(rgb[1, 0], [0, 255, 0])  # tumor

    def test_overlay_slice(self):
        from src.visualize import overlay_slice
        ct = np.random.randn(64, 64).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        result = overlay_slice(ct, mask)
        self.assertEqual(result.shape, (64, 64, 3))

    def test_overlay_slice_draws_kidney_outline_and_tumor(self):
        from src.visualize import KIDNEY_OUTLINE_COLOUR, overlay_slice
        ct = np.zeros((32, 32), dtype=np.float32)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:20, 8:20] = 1
        mask[20:23, 12:15] = 2
        result = overlay_slice(ct, mask, alpha=1.0)
        self.assertTrue((result[8, 10] == KIDNEY_OUTLINE_COLOUR).all())
        self.assertTrue(result[21, 13].any())

    def test_slice_qc_warning_flags_broad_mask(self):
        from src.visualize import _slice_qc_warning
        mask = np.ones((32, 32), dtype=np.uint8)
        self.assertEqual(_slice_qc_warning(mask), "QC: broad mask")

    def test_slice_title_includes_predicted_and_gt_tumor_counts(self):
        from src.visualize import _slice_title
        pred = np.zeros((8, 8), dtype=np.uint8)
        pred[1:3, 1:3] = 2
        volume = np.zeros((8, 8), dtype=np.uint8)
        volume[1:5, 1:5] = 2
        title = _slice_title(4, pred, volume)
        self.assertIn("tumor: 4 px", title)
        self.assertIn("GT: 16", title)

    def test_constrain_tumor_to_foreground(self):
        from src.visualize import constrain_tumor_to_foreground
        pred = np.zeros((4, 4), dtype=np.uint8)
        pred[0, 0] = 2
        pred[1, 1] = 2
        volume = np.zeros((4, 4), dtype=np.float32)
        volume[1, 1] = 1.0
        constrained = constrain_tumor_to_foreground(pred, volume)
        self.assertEqual(int(constrained[0, 0]), 0)
        self.assertEqual(int(constrained[1, 1]), 2)

    def test_render_montage(self):
        from src.visualize import render_montage
        tmpdir = tempfile.mkdtemp()
        vol = np.random.randn(16, 32, 32).astype(np.float32)
        pred = np.zeros((16, 32, 32), dtype=np.uint8)
        path = render_montage(vol, pred, "test_case", tmpdir, n_slices=4)
        self.assertTrue(os.path.isfile(path))
        shutil.rmtree(tmpdir)

    def test_render_montage_accepts_debug_marker_flag(self):
        from src.visualize import render_montage
        tmpdir = tempfile.mkdtemp()
        vol = np.ones((4, 16, 16), dtype=np.float32)
        pred = np.zeros((4, 16, 16), dtype=np.uint8)
        pred[:, 4:8, 4:8] = 2
        path = render_montage(vol, pred, "test_case_marker", tmpdir, n_slices=2, marker=True)
        self.assertTrue(os.path.isfile(path))
        shutil.rmtree(tmpdir)

    def test_visualize_case_writes_to_case_folder(self):
        import nibabel as nib
        from src.visualize import visualize_case

        tmpdir = tempfile.mkdtemp()
        try:
            cfg = _make_test_config(tmpdir)
            cfg_path = _write_config(tmpdir, cfg)
            case_dir = os.path.join(tmpdir, "case_00000")
            os.makedirs(case_dir)
            volume = np.zeros((4, 16, 16), dtype=np.float32)
            nib.save(
                nib.Nifti1Image(volume, affine=np.eye(4)),
                os.path.join(case_dir, "segmentation.nii.gz"),
            )
            os.makedirs(cfg["output"]["export"]["output_dir"])
            np.save(
                os.path.join(cfg["output"]["export"]["output_dir"], "case_00000_pred.npy"),
                np.zeros((4, 16, 16), dtype=np.uint8),
            )

            paths = visualize_case(config_path=cfg_path, case_id="case_00000")

            self.assertEqual(len(paths), 2)
            self.assertTrue(all(Path(p).parent.name == "case_00000" for p in paths))
        finally:
            shutil.rmtree(tmpdir)


# ────────────────────────────────────────────────────────────
# Tests — Benchmark
# ────────────────────────────────────────────────────────────

class TestBenchmark(unittest.TestCase):
    def test_benchmark_synthetic_cpu(self):
        from src.benchmark import benchmark_inference
        tmpdir = tempfile.mkdtemp()
        cfg = _make_test_config(tmpdir)
        cfg_path = _write_config(tmpdir, cfg)
        result = benchmark_inference(
            config_path=cfg_path,
            device_override="cpu",
            warmup_runs=1,
            timed_runs=2,
            synthetic=True,
        )
        self.assertIn("latency_ms", result)
        self.assertIn("throughput_fps", result)
        self.assertGreater(result["latency_ms"]["mean"], 0)
        shutil.rmtree(tmpdir)


# ────────────────────────────────────────────────────────────
# Tests — Config & Data Source
# ────────────────────────────────────────────────────────────

class TestLoadConfig(unittest.TestCase):
    def test_roundtrip(self):
        from src.pipeline import load_config
        tmpdir = tempfile.mkdtemp()
        cfg = _make_test_config(tmpdir)
        path = _write_config(tmpdir, cfg)
        loaded = load_config(path)
        self.assertEqual(loaded["pipeline"]["name"], "test")
        shutil.rmtree(tmpdir)


class TestDataSource(unittest.TestCase):
    def test_discover_case_ids_scans_input_folder(self):
        from src.pipeline import discover_case_ids
        tmpdir = tempfile.mkdtemp()
        try:
            for cid in ["case_00002", "case_00000", "case_00209"]:
                case_dir = os.path.join(tmpdir, cid)
                os.makedirs(case_dir)
                Path(case_dir, "segmentation.nii.gz").touch()
            os.makedirs(os.path.join(tmpdir, "notes"))

            self.assertEqual(
                discover_case_ids(tmpdir),
                ["case_00000", "case_00002", "case_00209"],
            )
        finally:
            shutil.rmtree(tmpdir)

    def test_discover_missing(self):
        from src.pipeline import DataSource
        ds = DataSource(root_dir="/nonexistent", case_ids=["case_00000"])
        paths = ds.discover()
        self.assertEqual(len(paths), 0)

    def test_discover_existing(self):
        from src.pipeline import DataSource
        tmpdir = tempfile.mkdtemp()
        case_dir = os.path.join(tmpdir, "case_00000")
        os.makedirs(case_dir)
        # create a dummy file
        fpath = os.path.join(case_dir, "seg.nii.gz")
        with open(fpath, "w") as f:
            f.write("dummy")
        ds = DataSource(root_dir=tmpdir, case_ids=["case_00000"], file_pattern="seg.nii.gz")
        paths = ds.discover()
        self.assertEqual(len(paths), 1)
        shutil.rmtree(tmpdir)


# ────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
