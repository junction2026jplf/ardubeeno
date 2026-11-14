# Training on Verda GPU servers

This page covers the GPU training on Verda (verda.com, formerly DataCrunch) on 2026-11-13/14: which servers were
rented, how a run went from the Mac to a GPU and back, and how the setup changed during the night and the next
day. The models and their scores are in [TRAINING.md](TRAINING.md) and [RESULTS.md](RESULTS.md). All times
are UTC (the hackathon venue is UTC+2).

## Why Verda, and why plain VMs

The first model trained on the Mac (M4 Pro, PyTorch MPS) at about 80 s per epoch. Early stopping ended that run
at epoch 33. Verda offers GPU VMs, serverless containers and batch jobs, Kubernetes/Slurm clusters, an LLM
inference API and storage. A single **on-demand GPU VM** was the right fit for these jobs:

- The models are small (YOLO11n/s, YOLO11n-cls; the largest was ConvNeXt-base), the data is 1–3 GB, and each run
  takes 5–60 minutes.
- On a VM, `make` and `scripts/train.py` run unchanged with `--device 0`.
- Batch jobs would need a Docker image with an HTTP `/health` endpoint, the weights returned through Object Storage
  and a hard deadline. Clusters are meant for multi-node work.
- Billing is in prepaid 10-minute blocks, and unused time is refunded, so short-lived instances are cheap.

Every instance was billed against a €200 prepaid balance. Effective prices were about 10% below list prices.

## The servers

| # | Hostname | GPU | vCPU / RAM | Location, contract | Effective price | Up (UTC) | What ran on it |
|---|---|---|---|---|---|---|---|
| 1 | `varroa-train` | A100 SXM4 40GB | 22 / 120 GB | FIN-02, on-demand | $1.52/h | Nov 13 22:18 – Nov 14 00:20 | Round 1 (`n320-long`, `n416-long`, `s320-long`); `n320-2cls`, `n320-os3-screen`, `n320-long-s1`; classifier seeds and the seed ensemble |
| 2 | `varroa-train-2` | A100 SXM4 40GB | 22 / 120 GB | FIN-02, on-demand | $1.52/h | 22:48 – 00:22 | Round 2 (`n320-os3`, `n320-screen`, `cls-n288`); `n320-long-s2`, `cls-s288`, `n320-tv-e40-screen`, `cls-n288-tv`, `cls3-n288`, `y26n-tv-e40-screen`; archive of instances 1 and 3 |
| 3 | (third instance) | L40S 48GB | 20 / 60 GB | FIN-02, on-demand | $1.78/h | 23:10 – 00:20 | `y26n320`, `y26s320`, `s416`, `cls-n384`, `cls-n288-s1` |
| 4 | `varroa-ev2` | A100 SXM4 40GB | 22 / 120 GB | FIN-02, on-demand | $1.52/h | 00:31 – 01:03 | External EV2 dataset: three classifier seeds with EV2 added |
| 5 | (short-lived) | A100 SXM4 40GB | 22 / 120 GB | on-demand | $1.52/h | about 01:08 – 01:24 | Measurements only: multi-frame aggregation and separation (AUC) on EV2 videos |
| 6 | `varroa-timm` | A100 SXM4 40GB | 22 / 120 GB | on-demand | $1.52/h | 06:43 – about 08:05 | Large pretrained classifiers (ConvNeXt-tiny/base, DINOv2-S), then `cls-n288-geo-s0/1/2` |
| 7 | `varroa-bees` | A100 SXM4 80GB | 22 / 120 GB | FIN-01, on-demand | $2.16/h | 09:57 – 10:59 | Bee detector (`bees-n640`, `bees-n416`, `bees-s640`); small classifiers `cls-n224`, `cls-n192`, `cls-n160` |
| 8 | `varroa-cam` | RTX PRO 6000 96GB | 30 / 90 GB | FIN-03, on-demand | $2.62/h | 11:10 – about 11:37 | Camera fine-tune: `det-cam-ft`, `det-cam-ft-tr`, `cls-cam-ft160`, `cls-cam-ft288` |
| 9 | `varroa-bees2` | A100 SXM4 40GB | 22 / 120 GB | FIN-01, **spot** | $0.77/h | 11:53 – 12:44 | `bees-n416-cam`, `bees-n416-flip`, `bees-n320-flip`; letterbox classifiers `cls-cam-sq160`, `cls-cam-sq288` |
| 10 | `varroa-synth` | A100 SXM4 40GB | 22 / 120 GB | FIN-01, **spot** | $0.77/h | 12:57 – 13:19 | `bees-n416-synth`, `bees-n416-synthft` |
| 11 | `varroa-neg` | RTX PRO 6000 96GB | 30 / 90 GB | FIN-03, on-demand | $2.62/h | 13:59 – 14:12 | Hard-negative bee detectors: `bees-n416-neg`, `bees-n416-camneg` |

**Cost:** the night (instances 1–3) came to about 4.8 GPU-hours and $7.55. By 06:39 the balance had fallen from
€200 to €190.78, after instances 4 and 5 as well. After the last instance it stood at €183.47, so the whole project
cost about **€16.5**.

**Which GPU, and when:** the first choice was the cheapest GPU with at least about 16 vCPUs. These models are small
enough that the CPU-side augmentation (mosaic, HSV, RandAugment) limits speed as much as the GPU does. The RTX A6000
and L40S were sold out at the start, so the A100 40GB (22 vCPU) became the default. When it was sold out, the order
of fallbacks was:

1. an A100 80GB;
2. a **spot** A100 40GB (half price, but it can be reclaimed);
3. an RTX PRO 6000 96GB.

The V100 was avoided: it has only 4 vCPUs, and the installed PyTorch probably no longer supports its compute
capability (sm70). The RTX 6000 Ada was avoided too, because it has only 10 vCPUs.

## How a run worked

All of this was done by an assistant thread with the Verda CLI (`brew install verda-cloud/tap/verda-cli`). No
script for it is checked in.

1. **Login and key, once:** `verda auth login` with an API client ID and secret (stored in `~/.verda/credentials`),
   then `verda ssh-key add` with the Mac's `~/.ssh/id_rsa.pub`.
2. **Find capacity:** `verda availability` (add `--spot` for spot) and `verda instance-types --gpu`.
3. **Create:**
   ```sh
   verda vm create --kind gpu --instance-type 1A100.40S.22V --location FIN-02 --contract pay_as_go \
     --os 24.04.cuda13.0 --os-volume-size 100 --hostname varroa-train --ssh-key <key-id> --wait
   ```
   On spot instances (`--is-spot`), `--os-volume-on-spot-discontinue delete_permanently` ensures a reclaimed
   instance doesn't leave a volume billing.
4. **Upload:** code, the `data/` splits (or `datasets/` for the bee detector) and any starting weights, over SSH as
   `root`. Symlinked datasets (`yolo/`, `cam_*`) were uploaded with links dereferenced, and every upload was checked
   with file counts and `md5`.
5. **Set up on the instance:** install `uv`, then `make setup convert` (Python 3.12 venv, PyTorch with CUDA); where
   needed, `uv pip install albumentations`. Fix the `path:` line in each dataset YAML to point to the instance's
   copy.
6. **Train:** each run in its own `tmux` window, so it survives a dropped SSH connection. Usually 2–3 runs share one
   GPU in parallel (`--batch 128 --workers 6 --cache ram`). The evaluations (`evaluate.py`, one at a time) are
   chained after training in a queue script.
7. **Copy back:** a watcher on the Mac copies each finished run to `runs/verda/<detect|classify|bees>/<name>/`
   (`weights/best.pt`, `weights/last.pt`, `results.csv`, `args.yaml`) and the eval logs to `runs/verda/evals/`.
   It then checks `best.pt` by md5 against the instance.
8. **Delete:** `verda vm delete <id> --with-volumes --yes`. Then poll `verda volume list` until the OS volume shows
   as detached, and delete it separately if it is still listed. The job is finished only when both
   `verda vm list` and `verda volume list` are empty.

## How it evolved

### Night 1: three A100-class instances, a tuning tournament (22:18 – 00:22)

The goal was to tune the mite models until we were sure we had hit a wall. A second assistant thread (on the Mac)
proposed each round's experiments; this thread ran them and sent the results back.

- **Round 1 (instance 1):** three runs in parallel on one A100 (`n320-long`, `n416-long`, `s320-long`) with no
  early stopping, AdamW 1e-3 cosine, 100 epochs. Each took about 21 s per epoch with three sharing the GPU (Mac:
  80 s). The GPU stayed at 100% and the CPUs at about 50%, so none of the runs was waiting on data.
- **Scale-out:** the user allowed more instances, so instance 2 (another A100) and instance 3 (an L40S, since the
  A100 had sold out) followed within an hour. Instance 2's data was copied from instance 1 inside the datacenter,
  which was much faster than uploading again from the Mac.
- **What the night found:** every detector variant landed at 0.877–0.888 per-bee test F1, within seed noise
  (±0.006). The variants were longer training, 416 px, YOLO11s, YOLO26n/s, under-wing oversampling, a two-class
  detector and screen augmentation. The one real gain was a per-bee classifier (`cls-n288`) fused with the
  detector, at about 0.90. Both threads agreed at 00:16 that the model had hit a wall. All three instances were
  deleted by 00:22, after everything was archived on instance 2 and the final models were copied to the Mac.

### Night 2: short single-purpose instances (00:31 – 01:24)

After the shutdown the local thread proposed one more idea, external data (the EV2 dataset). From then on each
experiment got **its own short-lived instance**, created for one job and deleted when it was done:

- `varroa-ev2`: the EV2 dataset was downloaded straight to the instance, because the Mac's disk was full. Adding
  EV2 lowered the classifier's score (0.885 vs 0.893).
- A second short A100: measurements only (multi-frame aggregation, AUC on unseen footage), about 15 minutes.

### Morning: one instance per request, cheaper GPUs (06:43 – 14:12)

From the morning on, the jobs came as separate requests from the project threads: the large-classifier check, the
bee detector, small classifiers for speed, the camera fine-tune, synthetic and hard-negative bee data. The user gave
standing approval to use Verda for training. Each request got its own instance, which was deleted as soon as its
runs were copied back. A100 40GB on-demand capacity became scarce, so the choice of GPU changed:

- 09:57: A100 80GB in FIN-01 (no 40GB free). The 80 GB let three 640 px bee-detector runs share the GPU.
- 11:10 and 13:59: RTX PRO 6000 96GB in FIN-03, the cheapest free GPU with enough vCPUs.
- 11:53 and 12:57: **spot** A100 40GB in FIN-01 at $0.77/h, half the on-demand price. Neither was reclaimed.

A standing rule was added at 12:11: when more than two independent jobs arrive, spread them across several
instances so they run fully in parallel, instead of packing them onto one GPU.

## Problems hit, and the fixes

| Problem | Effect | Fix |
|---|---|---|
| macOS's built-in `rsync` (openrsync) reported success but left old scripts on the instance | Round 1 trained with an older `train.py` (val NMS IoU 0.7 instead of 0.45), so its `best.pt` was picked under different rules | Copy code with `scp`; compare `md5` of `scripts/*.py` on Mac and instances before each round |
| Queue script waited with `pgrep -f scripts/train.py`, which also matched the tmux server's command line | Instance 1 sat idle for about 15 minutes | Match the actual Python process, not the command string |
| Four evaluations at once on one GPU | Six crashed with CUDA out-of-memory | Evaluations run one at a time per machine and retry on OOM |
| Two classifier runs with `--cache ram` on the L40S (60 GB RAM, against 120 GB on the A100s) | The kernel's OOM killer took down the tmux session and the running evaluations | `--cache disk` on low-RAM instances |
| Mac disk full (about 200 MB free) at 23:58 | Copies back to the Mac stopped | Archived instances 1 and 3 onto instance 2, then copied only the final models and logs (21 MB plus 1 MB) with a free-space check before each file |
| `api.verda.com` showed a Cloudflare "domain suspended" page at 00:20 | CLI and console stopped working | Use the old DataCrunch API: `verda --base-url https://api.datacrunch.io/v1 ...` |
| RTX PRO 6000 (confidential-computing image) came up as GPU "not ready" | CUDA failed | `nvidia-smi conf-compute -srs 1` before training |
| OS volume still listed as "attached" right after `vm delete` | It would keep billing if left | Poll `verda volume list`, delete it once it is detached |
| `best.pt` exists while training is still running | An early EV2 evaluation scored half-trained models | Start evaluations only after the run has finished |
| Ultralytics didn't follow a converter's directory symlinks | EV2 val and test came up with 0 images | Copy the folders instead of linking them |
| Datasets with per-file symlinks into `captures/` | A plain copy would upload dangling links | Upload with links dereferenced, then check file counts on the instance |
