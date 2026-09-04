# DeepSpeed ZeRO-1 — DistilBERT on AG News (2×T4)

Self-contained Kaggle notebook: fine-tune **`distilbert-base-uncased`** on a
6 000-sample slice of **AG News** with **DeepSpeed ZeRO-1**, on **2×NVIDIA T4
(15 GB each)** via `accelerate launch --num_processes 2`.

## What runs inside the notebook

| Item                 | Value                                            |
|----------------------|--------------------------------------------------|
| Model                | `distilbert-base-uncased` (4-class head)         |
| Data                 | `ag_news` → 6 000 train / 2 000 test             |
| Precision            | FP16 (T4 has no bf16)                            |
| Batch / GPU          | 32                                               |
| Gradient accum.      | 1                                                |
| Effective batch      | 64                                               |
| Train length         | 300 steps                                        |
| Optimizer            | AdamW, states sharded 2-way (ZeRO-1)             |
| Launch               | `accelerate launch --num_processes 2`            |

## How to run (Kaggle)

1. Upload this file to your Kaggle account (or open it directly if it's there).
2. Accelerator: **T4** (or **P100** — also 2× available).
3. GPU count: **2**.
4. **Run All**. Everything else (pip installs, data download, training,
   post-train probe) is inside the notebook.

## How to run (local, 2 GPUs)

```bash
pip install "transformers==4.57.0" "datasets" "evaluate" \
            "accelerate" "deepspeed"
# 1. Copy the two %%bash cells (ds_zero1.json, train_ds.py) out of the notebook
#    and save them in a working dir.
# 2. Launch:
accelerate launch --num_processes 2 --mixed_precision fp16 \
  train_ds.py --out_dir out_ds --per_device_train_bs 32 \
              --steps 300 --ds_config ds_zero1.json
```

## What to look for in the log

After `trainer.train()` finishes, the training script runs a one-shot probe
and dumps `metrics.json`. A successful run prints, on each rank:

```
[rank 0/2] / [rank 1/2]
[check] deepspeed_enabled=True  v=0.19.6
  …
  [proof] engine=DeepSpeedEngine  is_DeepSpeedEngine=True  is_DDP=False
  [proof] optimizer=DeepSpeedOptimizerWrapper
  {'world_size': '2', 'zero_stage': 1,   'engine_cls': 'DeepSpeedEngine',
   'optimizer_cls': 'DeepSpeedOptimizerWrapper', 'steps/sec': <float>,
   'final_acc': <float>}
[result] { … same dict … }
```

`is_DeepSpeedEngine=True` and `is_DDP=False` are the direct evidence that
the trainer is wrapped in DeepSpeed's `DeepSpeedEngine`, not `DistributedDataParallel`.

## Notes

- **T4 is sm_75** → no bfloat16, no `fusedAdam` from the DeepSpeed JIT
  build. ZeRO-1 uses standard `torch.optim.AdamW`, so neither matters.
- **No NVLink** on most 2×T4 boxes; NCCL uses PCIe. The notebook sets
  `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` defensively.
- **`train_batch_size` is hardcoded** in `ds_zero1.json` to 64. When launched
  through `accelerate launch`, DeepSpeed 0.19.x does not infer it from
  `per_device_train_batch_size × N × accum`.

## Upgrade path (in `ds_zero1.json`)

- ZeRO-1 → ZeRO-2: `"stage": 1` → `"stage": 2`
- ZeRO-2 → ZeRO-3: `"stage": 3`
- Bigger model: swap `distilbert-base-uncased` for `bert-base-uncased`
- More data: raise `range(6000)` in the train slice
