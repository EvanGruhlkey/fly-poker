# The trained fly in the poker app

The target is the core method in [fly-chess](https://github.com/cesp99/fly-chess), pinned for this adaptation at commit `3c83567dfb6fa71cecc1556c75f17c8bb38fd5cf`. Poker replaces the chess observation, legal moves, teacher and game rewards.

## Model

The FlyWire graph determines which connections exist and the sign of each connection. Training changes connection magnitudes through `sign * softplus(gain)`, neuron biases and leak rates, the sensory projection and the motor policy/value heads. The public table state and the fly's own cards enter sensory neurons. Actions are read from descending and motor neurons.

Upstream supports several rate activations. This adaptation uses its saturating rectified form to bound recurrent activity. Connection magnitudes start from `log1p(synapse_count)` scaled by the mean incoming magnitude across neurons. This differs from the first frozen-weight sigmoid pilot in the historical experiment reports.

## Poker training

Imitation uses legal decisions from an equity-based poker teacher. The teacher samples unknown cards rather than reading the actual opponent cards or future deck. This is a generated baseline, unlike fly-chess's human Lichess training data.

The second stage uses sampled poker trajectories and terminal net winnings against mixed opponents, including a frozen snapshot of the fly. Chess PUCT search is not copied into poker because it would require a strategy for imperfect information. These stages follow the imitation-then-self-play structure, but do not reproduce the scale or algorithm of the chess project.

Whole-hand training and evaluation seeds are separate. Evaluation swaps player positions on paired decks and reports sample-limited bootstrap intervals. A successful optimizer run does not establish a strong poker strategy or an advantage from biological wiring.

## Browser execution

A persistent Web Worker runs the exported sparse network locally. It receives only the fly's own cards, public board, stacks and legal betting state. The model selects one of seven legal actions. The game engine applies that action and retains ownership of the cards, betting rules and chip animations.

Python and JavaScript must agree on encoded observations, action masks, recurrence outputs and policy logits. Resetting a match invalidates pending responses. Model load failures must appear in the app rather than silently replacing the network with a heuristic bot.

The FlyWire-derived graph and model artifacts are [CC BY-NC 4.0](https://huggingface.co/cesp99/fly-chess/blob/main/README.md). The source data checksum and attribution are in `training/data_manifest.json`. Upstream code is MIT licensed.

## Delivered model

The [published v2 model](https://github.com/EvanGruhlkey/fly-poker/releases/tag/brain-v2-seed17) completed 256 imitation steps and 256 learning updates on a Modal NVIDIA L4: 2,048 hands, including 1,018 frozen-opponent self-play hands. Install with `npm --prefix web run brain:fetch`.

The full exported graph matched Python on 12 positions with maximum logit error below 0.000001. The actual browser worker selected matching actions in all 12 cases and took 72–88 ms per inference. All 47 Python tests and 35 web tests passed, as did the production build.

The small paired evaluation returned +10.665 BB/hand against random play and -14.465 BB/hand against the equity baseline. This is an experimental policy, not evidence of strong poker play. [The measured report](../experiments/brain-v2-training.json) includes confidence intervals and limitations.
