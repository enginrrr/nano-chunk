# RunPod CTF Comparison

This checkout keeps upstream `train_gpt.py` untouched and adds `train_gpt_ctf.py` for the Chunk Transition Flow test.

## Setup

```bash
git clone https://github.com/KellerJordan/modded-nanogpt.git
cd modded-nanogpt
pip install -r requirements.txt
python data/cached_fineweb10B.py 9
```

If using this local working tree instead of a fresh clone, copy `train_gpt_ctf.py` and `run_ctf.sh` into the RunPod checkout.

## Baseline

```bash
./run.sh
```

## Chunk Transition Flow

```bash
./run_ctf.sh
```

## What Changed

`train_gpt_ctf.py` adds the local best CTF primitive in pre/post residual lanes:

```text
m_i = inclusive causal 16-token chunk prefix
r_s = 0.75*r1 + 0.25*r2
z_i = x_i + sigmoid(a)*(m_i-x_i) + sigmoid(b)*(m_i-r_s)
T(x_i) = W2(ReLU(W1 z_i))^2
```

It uses separate CTF weights for each layer and placement. The output projection is zero-initialized, `chunk_gate` starts at `0`, `transition_gate` starts at `-6`, and each pre/post CTF residual starts behind a learned output gate scaled by `0.05 * sigmoid(gate)` so the new path begins at 2.5% strength while preserving the chunk transition formula.

## Comparison Rule

Compare the final `val_loss` and total training time from each log in `logs/`. The first run on a fresh machine pays compile cost, so run baseline and CTF in the same environment after warmup or run each twice if budget allows.
