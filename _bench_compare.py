"""Quick compiled vs eager benchmark on the same volume."""
import time, torch, yaml, warnings, copy
warnings.filterwarnings("ignore")
from src.pipeline import InferenceOperator, HoloscanInputOperator, PreProcessingOperator, resolve_device

cfg = yaml.safe_load(open("config.yaml"))
dev = resolve_device("xpu")

from pathlib import Path
inp_op = HoloscanInputOperator(cfg)
pre_op = PreProcessingOperator(cfg, dev)
nifti = Path(cfg["data"]["root_dir"]) / "case_00030" / "segmentation.nii.gz"
vol = inp_op.read_nifti(nifti)
tensor = pre_op(vol)

# --- Compiled ---
cfg_c = copy.deepcopy(cfg)
cfg_c["inference"]["triton_backend"] = True
op_c = InferenceOperator(cfg_c, dev)
# warmup (includes JIT)
_ = op_c(tensor); torch.xpu.synchronize()
times_c = []
for _ in range(3):
    t0 = time.perf_counter()
    _ = op_c(tensor); torch.xpu.synchronize()
    times_c.append(time.perf_counter() - t0)

# --- Eager ---
cfg_e = copy.deepcopy(cfg)
cfg_e["inference"]["triton_backend"] = False
op_e = InferenceOperator(cfg_e, dev)
_ = op_e(tensor); torch.xpu.synchronize()
times_e = []
for _ in range(3):
    t0 = time.perf_counter()
    _ = op_e(tensor); torch.xpu.synchronize()
    times_e.append(time.perf_counter() - t0)

print(f"Compiled: {[round(t,3) for t in times_c]}  avg={sum(times_c)/3:.3f}s")
print(f"Eager:    {[round(t,3) for t in times_e]}  avg={sum(times_e)/3:.3f}s")
speedup = (sum(times_e)/3) / (sum(times_c)/3)
print(f"Speedup (compiled/eager): {speedup:.2f}x")
