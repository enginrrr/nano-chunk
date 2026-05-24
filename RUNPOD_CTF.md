# RunPod CTF One-Shot Notes

This repo keeps upstream `train_gpt.py` as the baseline and uses `train_gpt_ctf.py` for the Chunk Transition Flow test.

## Known-Valid Environment

The valid baseline run used:

```text
PyTorch 2.10.0+cu128
CUDA 12.8
Triton 3.6.0
```

Fresh RunPod images may start with older Torch/Triton. If `triton.tools.tensor_descriptor` is missing, the run will fail before training. Fix the environment before launching:

```bash
python3 -m pip install --upgrade --no-cache-dir torch==2.10.0 triton==3.6.0
python3 -m pip install --no-cache-dir tqdm tiktoken datasets huggingface-hub kernels==0.9.1 setuptools typing-extensions==4.15.0
python3 - <<'PY'
import torch, triton
from triton.tools.tensor_descriptor import TensorDescriptor
print(torch.__version__, torch.version.cuda)
print(triton.__version__, TensorDescriptor)
PY
```

Do not launch if that import check fails.

## Data

The FineWeb cache for `python3 data/cached_fineweb10B.py 9` should contain:

```text
data/fineweb10B/fineweb_val_000000.bin
data/fineweb10B/fineweb_train_000001.bin ... fineweb_train_000009.bin
```

There is no required `fineweb_train_000000.bin` for this run. Check with:

```bash
ls data/fineweb10B/fineweb_train_*.bin | wc -l
ls data/fineweb10B/fineweb_val_*.bin | wc -l
```

Expected counts are `9` train files and `1` val file.

## Required Preflight

Run this before every launch:

```bash
cd /workspace/nano-chunk
git fetch origin master
git reset --hard cf13ddd7f0afafaec3d81a2ac4d131b3dacdd602

python3 -m py_compile train_gpt_ctf.py train_gpt.py

grep -c "out_safe = torch.nan_to_num" train_gpt_ctf.py
grep -c "if i in (3, 7):" train_gpt_ctf.py
grep -c "self.ctf_gates\[:, 0, :\].fill_(-3.0)" train_gpt_ctf.py
grep -c "self.ctf_gates\[:, 2, :\].fill_(-2.0)" train_gpt_ctf.py
grep -c "non-finite validation loss" train_gpt_ctf.py
grep -c "^def print_ctf_stats" train_gpt_ctf.py
grep -c "# H8 defense:" train_gpt_ctf.py
grep -c "# H10 defense:" train_gpt_ctf.py
grep -c "model.ctf_bank.data = model.ctf_bank.data.bfloat16" train_gpt_ctf.py
grep -c "nan_to_num.*\.clamp_(" train_gpt_ctf.py
grep -c "self.ctf_gates\[:, 0, :\].zero_()" train_gpt_ctf.py
grep -c "self.ctf_gates\[:, 2, :\].zero_()" train_gpt_ctf.py
grep -c "p_cfg.label.startswith(\"ctf_\")" train_gpt_ctf.py
grep -c "p_cfg.label == \"ctf_bank\"" train_gpt_ctf.py
grep -c "F.relu(y).clamp(max=32.0).square()" train_gpt_ctf.py
grep -c "c_fc.float()" train_gpt_ctf.py

ps -eo cmd | grep -E 'torchrun|train_gpt_ctf.py|train_gpt.py' | grep -v grep
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

Expected grep counts, in order:

```text
1 1 1 1 1 1 1 1 0 0 0 0 1 1 1 1
```

The process check must return no training process. Every GPU memory value must be `0`.

Reset compile caches before launch:

```bash
rm -rf /workspace/tmp /workspace/torchinductor_cache /workspace/triton_cache
mkdir -p /workspace/tmp /workspace/torchinductor_cache /workspace/triton_cache
```

## Launch

Use proxy SSH only. Do not use direct TCP for this pod.

Use `--run_id`; do not rely on `NANOGPT_RUN_ID`.

```bash
RUN_ID=ctf-one-shot-cf13ddd-$(date +%Y%m%d-%H%M%S)
TORCHINDUCTOR_CACHE_DIR=/workspace/torchinductor_cache \
TRITON_CACHE_DIR=/workspace/triton_cache \
TMPDIR=/workspace/tmp \
torchrun --standalone --nproc_per_node=8 train_gpt_ctf.py --run_id "$RUN_ID" \
  2>&1 | tee "/workspace/${RUN_ID}.console.log"
```

## First Checkpoint

At step 0, required signals:

```text
val_loss finite near 10.83
c_proj_rms:0.000000
c_proj_absmax:0.000000
chunk_gate_mean near 0.047
transition_gate_mean near 0.0025
out_gate_mean near 0.006
```

If step 0 is wrong, kill immediately.

At step 250, `val_loss` must be finite and `c_proj_rms` must be greater than zero for CTF to count as active.
