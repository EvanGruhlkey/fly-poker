# Fly Poker

Heads-up no-limit Texas Hold’em against a fruit fly in a full-screen casino. Both players begin with 100 BB. Blinds are 0.5 / 1 BB, stacks carry between hands, and the dealer button alternates.

The browser game includes shuffled cards, legal betting, all-ins, unmatched bet refunds, best-five-of-seven showdowns, split pots, and animated chips, dealing, reveals, and payouts. The opponent runs a trained FlyWire-derived neural network locally in a browser worker. It was trained on a Modal NVIDIA L4 using equity-teacher imitation followed by self-play. See [the model and measured results](docs/brain-in-app.md).

## Run the game

Requires Node.js 20 or later.

```sh
cd web
npm install
npm run brain:fetch
npm run dev
```

```sh
npm test
npm run build
```

## Python research backend

Requires Python 3.12 or later and uv.

```sh
uv sync
uv run pytest
```

The original `fly_poker/connectome` modules are a prototype. The MaleCNS manifest contains an unresolved dataset checksum and must not be treated as a verified dataset or a trained model.

The first training experiment uses the checksum-pinned FlyWire graph published with the reference project. See [the research plan](docs/training-research.md) for provenance, compute limits, evaluation requirements, and scientific limitations.

## Credits and licenses

The casino design is inspired by [Carlo Esposito’s fly-chess](https://github.com/cesp99/fly-chess), whose code and fly artwork are MIT licensed. The copied artwork license is retained in `web/public/assets/fly-chess-LICENSE.txt`.

Playing-card SVGs are by [Adrian Kennard](https://www.me.uk/cards/) and are CC0. Their attribution is retained with the card assets.

The FlyWire graph is CC BY-NC 4.0 and requires attribution and non-commercial use. Those terms apply to that graph and derived model artifacts; do not infer a permissive data license from the reference code’s MIT license. Raw connectome data and trained weights are excluded from Git. Checkpoints are stored in the Modal `fly-poker-artifacts` volume and can be downloaded locally.

## Train on Modal

Install and authenticate the Modal CLI, then verify your workspace credits and hard usage cap before launching. This account currently has $1 monthly credit; GPU access is checked separately.

```sh
modal run modalapp.py --mode smoke --accelerator gpu
modal run modalapp.py --mode train --accelerator gpu --seed 17 --warm-steps 256 --updates 256 --wall-seconds 900
```

GPU dispatch requests one L4, two CPU cores and 8 GiB RAM, with no retries and a 20-minute worker timeout. The training budget is capped at 15 minutes and reserves evaluation time. Use `--accelerator cpu` for CPU runs. Resume requires an existing checkpoint with matching seed and batch.

The first CPU pilot completed 64 warm-up steps and 32 actor-critic updates over 256 hands. Its small held-out evaluation was inconclusive. See [the measured pilot](experiments/cpu-pilot.json). These historical v1 weights are superseded by the trained v2 browser model.

An L4 GPU continuation completed another 32 updates, reaching 512 total training hands in 39.4 seconds for the continuation and evaluation. Saved weights changed after resuming. [GPU results](experiments/gpu-pilot.json) remain sample-limited and do not establish poker strength.

The current v2 run completed 256 imitation steps and 256 learning updates over 2,048 hands, including 1,018 frozen-opponent self-play hands. [Measured results](experiments/brain-v2-training.json) include the small held-out evaluation and browser parity checks. The model remains experimental and lost to the equity baseline. Download the checksum-pinned weights with `npm run brain:fetch`; artifacts are published in the [model release](https://github.com/EvanGruhlkey/fly-poker/releases/tag/brain-v2-seed17).
