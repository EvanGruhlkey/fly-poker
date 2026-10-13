# Training and evaluation

## Python research backend

Requires Python 3.12 or later and uv.

```sh
uv sync
uv run pytest
```

The original `fly_poker/connectome` modules are a prototype. The MaleCNS manifest contains an unresolved dataset checksum and must not be treated as a verified dataset or a trained model.

The first training experiment uses the checksum-pinned FlyWire graph published with the reference project. See [the research plan](../docs/training-research.md) for provenance, compute limits, evaluation requirements, and scientific limitations.

## Credits and licenses

The casino design is inspired by [Carlo Esposito’s fly-chess](https://github.com/cesp99/fly-chess), whose code and fly artwork are MIT licensed. The copied artwork license is retained in `web/public/assets/fly-chess-LICENSE.txt`.

Playing-card SVGs are by [Adrian Kennard](https://www.me.uk/cards/) and are CC0. Their attribution is retained with the card assets.

The FlyWire graph is CC BY-NC 4.0 and requires attribution and non-commercial use. Those terms apply to that graph and derived model artifacts; do not infer a permissive data license from the reference code’s MIT license. Raw connectome data and trained weights are excluded from Git. Checkpoints are stored in the Modal `fly-poker-artifacts` volume and can be downloaded locally.

## Train on Modal

Install and authenticate the Modal CLI, then verify your workspace credits and hard usage cap before launching. Read `modal billing summary --json` and `modal billing rates --json`; check remaining credit and the hard spend limit in the dashboard when the CLI does not expose them.

```sh
modal run modalapp.py --mode smoke --accelerator gpu
modal run modalapp.py --mode train --accelerator gpu --seed 17 --warm-steps 256 --updates 256 --wall-seconds 900
```

GPU dispatch requests one L4, two CPU cores and 8 GiB RAM, with no retries and a 20-minute worker timeout. The training budget is capped at 15 minutes and reserves evaluation time. Use `--accelerator cpu` for CPU runs. Resume requires an existing checkpoint with matching seed and batch.

The first CPU pilot completed 64 warm-up steps and 32 actor-critic updates over 256 hands. Its small held-out evaluation was inconclusive. See [the measured pilot](../experiments/cpu-pilot.json). These historical v1 weights were superseded by v2 and later v4 models.

An L4 GPU continuation completed another 32 updates, reaching 512 total training hands in 39.4 seconds for the continuation and evaluation. Saved weights changed after resuming. [GPU results](../experiments/gpu-pilot.json) remain sample-limited and do not establish poker strength.

The earlier v2 run completed 256 imitation steps and 256 learning updates over 2,048 hands, including 1,018 frozen-opponent self-play hands. [Measured results](../experiments/brain-v2-training.json) include the small held-out evaluation and browser parity checks. The model remains experimental and lost to the equity baseline. Download the checksum-pinned weights with `npm run brain:fetch`; artifacts are published in the [model release](https://github.com/EvanGruhlkey/fly-poker/releases/tag/brain-v2-seed17).

## Improve and evaluate poker strength

The experimental v4 pipeline adds versioned 267-feature observations, cached supervised labels, adapter and constrained graph stages, clipped PPO, persistent historical opponents, and held-out tournaments. [The audit](../docs/poker-strength-audit.md) records the methods and measured failures. The new teacher is not yet validated, and the v4 pilot did not establish better Holdem play. The deployed game now uses the v4 seed 43 imitation checkpoint, with 1,024 imitation updates and no PPO updates. Its playing strength remains unproven.

Benchmark a candidate teacher before enabling its supervision.

```sh
uv run python -m training.teacher_benchmark --pairs 128 --seed 9417 --samples 32 --seconds 180 --output experiments/teacher-validation.json
```

Generate reusable heuristic imitation labels. These reproduce the existing equity heuristic with legal smoothing; they do not claim a stronger teacher. Use `--teacher rollout` only for teacher research until its benchmark gate passes. The historical `equity` soft-EV labels encourage oversized bets and remain a debugging option.

```sh
uv run python -m training.dataset --output runs/heuristic-train.npz --seed 31 --count 4096 --samples 64 --split train --teacher heuristic --seconds 300
uv run python -m training.dataset --output runs/heuristic-validation.npz --seed 31 --count 512 --samples 64 --split validation --teacher heuristic --seconds 120
```

After checking account limits, run the L4 smoke and a 60-second pilot. The experiment JSON names a fresh output directory and the existing v2 migration source in the Volume.

```sh
modal run modalapp.py --mode smoke --accelerator gpu --batch 4 --iterations 1
modal run modalapp.py --mode train --accelerator gpu --experiment-config experiments/pilot-v4.json --wall-seconds 60
```

For a continuation, increase `updates` in a copy of the experiment JSON and pass `--resume`. Keep its run name, seed, batch, migration source, stage settings, and datasets unchanged. A fresh run refuses to overwrite an existing experiment directory. Copying the pilot JSON without changing its run name resumes the saved pilot only when `--resume` is supplied.

To use cached labels on Modal, upload both NPZ files to the Volume and set `dataset` and `validation_dataset` in the experiment JSON to their `/artifacts/` paths. Rollout labels additionally require `teacher_report` with passing confidence bounds, matching teacher source SHA, and matching samples.

Local training uses the same configuration through `uv run python -m training.train --graph data/flywire/full.npz --config YOUR_CONFIG.json --output runs/YOUR_RUN --seconds 60`. Use local checkpoint and dataset paths in that JSON.

Download checkpoints and run held-out debugging comparisons. Substitute an available previous checkpoint for the optional `--previous` argument.

```sh
modal volume get fly-poker-artifacts /runs/brain-v2-seed-17/latest.pt runs/v2.pt
modal volume get fly-poker-artifacts /runs/brain-v4-pilot-seed-23/latest.pt runs/v4.pt
uv run python -m training.evaluate_cli --graph data/flywire/full.npz --candidate runs/v4.pt --starting runs/v2.pt --output experiments/v4-validation.json --pairs 8 --seconds 180 --equity-samples 16
```

For final comparisons use `--domain test --pairs 1000`, a sufficient bounded evaluation time, and `--previous PATH` for historical opponents. Reports include BB per hand, BB/100, bootstrap intervals, seat and stack summaries, action frequencies, aggression, and showdowns. Incomplete evaluations cannot pass promotion. The CLI deliberately does not attest export parity or publish models.

Run the separate Kuhn research experiment without a GPU.

```sh
uv run python -m training.kuhn_experiment --hands 2000 --seconds 30 --output experiments/kuhn-nfsp-pilot.json
```

The authorized follow-up uses the saved round-two configurations and a batched L4 evaluator:

```sh
modal volume put fly-poker-artifacts runs/heuristic-train.npz /datasets/heuristic-train.npz
modal volume put fly-poker-artifacts runs/heuristic-validation.npz /datasets/heuristic-validation.npz
modal run modalapp.py --mode train --accelerator gpu --experiment-config experiments/round2-ppo.json --wall-seconds 300
modal run modalapp.py --mode train --accelerator gpu --experiment-config experiments/round2-imitation.json --wall-seconds 300
modal run modalapp.py --mode evaluate --candidate /artifacts/runs/brain-v4-ppo-seed-31/latest.pt --output /artifacts/runs/brain-v4-ppo-seed-31/validation-128.json --eval-pairs 128 --seed 731 --samples 16 --wall-seconds 600
modal run modalapp.py --mode controls --steps 64 --wall-seconds 90 --seed 31
```

Limit concurrency and account for aggregate compute across jobs. Pass `--resume` for an existing training directory or a resumable batched evaluation report. Evaluations pin checkpoint hashes, source hashes, deal manifests and sampling controls. Architecture controls retrain from scratch on shared batch-sequence prefixes and wall limits; they do not establish equal FLOPs or a biological advantage. Use `--mode controls --steps 100000 --wall-seconds 90` for the wall-budget experiment, followed by `--mode control-evaluate --eval-pairs 32 --wall-seconds 120 --seed 31`.

The coverage census found too few decisions facing bets in the original cache. Generate the pressure-balanced variant with `--sampling balanced-pressure` and use the saved `experiments/round2-pressure.json` configuration after uploading the new files to `/datasets/pressure-train.npz` and `/datasets/pressure-validation.npz`. The sampler balances street and pot-odds strata on legal hands; unfiltered heldout tournaments measure actual performance.

The compact teacher is a separate CPU research tool: `uv run python -m training.compact_teacher --seconds 60 --seed 31 --output runs/compact-seed-31`. It failed its equity validations and is not used for fly supervision. [Round-two results](../docs/round2-results.md) contain measured comparisons. Regenerate that table with `uv run python -m experiments.summarize_round2`; plot completed training updates with `uv run --with matplotlib python experiments/plot_learning.py`.

Training writes frequent atomic local checkpoints and synchronizes the Modal Volume at most once per minute during normal updates, plus forced synchronization at job boundaries. A worker crash can lose work since the last remote synchronization. A normal budget stop restores a completed local checkpoint and synchronizes it. Reports include persistence timing for new runs.
