# Fly Poker

Heads-up no-limit Texas Hold’em against a fruit fly in a full-screen casino. Both players begin with 100 BB. Blinds are 0.5 / 1 BB, stacks carry between hands, and the dealer button alternates.

The browser game includes shuffled cards, legal betting, all-ins, unmatched bet refunds, best-five-of-seven showdowns, split pots, and animated chips, dealing, reveals, and payouts. The currently shipped opponent is a local heuristic bot. Neural training is a separate research experiment and is not yet connected to the game.

## Run the game

Requires Node.js 20 or later.

```sh
cd web
npm install
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
modal run modalapp.py --mode train --accelerator gpu --seed 7 --warm-steps 64 --updates 64 --resume --wall-seconds 900
```

GPU dispatch requests one L4, two CPU cores and 8 GiB RAM, with no retries and a 20-minute worker timeout. The training budget is capped at 15 minutes and reserves evaluation time. Use `--accelerator cpu` for CPU runs. Resume requires an existing checkpoint with matching seed and batch.

The first CPU pilot completed 64 warm-up steps and 32 actor-critic updates over 256 hands. Its small held-out evaluation was inconclusive. See [the measured pilot](experiments/cpu-pilot.json). The browser opponent still uses the local heuristic.
