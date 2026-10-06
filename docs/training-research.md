# Training research and first experiment

Research checked on 2026-10-05. This document distinguishes a planned experiment from measured results.

## Data

Use the [reference project’s released BrainGraph](https://huggingface.co/cesp99/fly-chess/blob/main/README.md), derived from FlyWire Codex snapshot 783. It contains 134,209 selected neurons and 2,700,513 neuron-pair connections. Synapse counts are connection weights, not separate graph edges.

Pin Hugging Face revision `cad733198a14d76a35149ecbe5f68bc6efa97732`, file `graph/full.npz`, size 39,910,529 bytes, and SHA-256 `4d63295f5e0d9b9ca7b4cf7802f975c2fc9fad87f29ee451305ccab5de02abe2`. The digest was checked against the hosting provider’s LFS metadata. A fetcher must verify the downloaded bytes before loading them with `allow_pickle=False`.

The release explicitly uses CC BY-NC 4.0. Retain attribution to Carlo Esposito, the FlyWire consortium, and the source papers. Model artifacts derived from this graph are for non-commercial use.

[MaleCNS](https://male-cns.janelia.org/download/) remains a future option under its separately stated data terms. The original project manifest confuses synapses with neuron-pair connections and has no verified checksum. The current synchronous prototype also does not establish elapsed biological milliseconds.

## Model

Use a sparse, signed, leaky rate network on the measured FlyWire topology. Preserve the connection pattern and signs. Structured own-hole-card, public-board, position, street, stack, and betting features enter through the graph’s sensory input indices. Policy and value readouts consume graph output neurons, with no direct observation-to-policy bypass.

Sparse matrix multiplication must replace the prototype’s batch-by-edge gather. Eight recurrent steps are an initial engineering setting, not eight biological milliseconds. Normalize connection weights to stabilize propagation. The first budget-limited experiment trains input and output projections through a fixed graph. This differs from the reference project, which also learns constrained synaptic magnitudes and other parameters.

## Learning and evaluation

Warm-start value predictions from Monte Carlo equity over unknown cards sampled without consulting the opponent’s actual hole cards or the future deck. An equity-threshold policy is a baseline, not an optimal poker teacher. Do not provide teacher equity as a network input.

A small actor-critic pilot against mixed and frozen opponents can test whether learning and checkpointing work. It does not establish equilibrium strategy. [NFSP](https://arxiv.org/abs/1603.01121) and [Deep CFR](https://arxiv.org/abs/1811.00164) provide stronger theoretical approaches for imperfect-information games and inform later work.

Require whole-hand-disjoint training and evaluation seeds, legal action masks, exact chip conservation, and rewards measured as net BB after accounting for blinds once. Specify the finite action abstraction used by the model.

Evaluate position-swapped paired deals against random legal actions, check/call, and an equity baseline. Report BB per hand or BB/100 with a confidence interval. A positive lower confidence bound against one baseline establishes performance only against that baseline.

Zeroed and shuffled graph interventions should report output changes and gameplay changes. These show dependence on the graph. Claims of a biological topology advantage require fairly retrained controls and a direct-observation model with comparable capacity and training budget.

## Modal budget

[Modal Starter](https://modal.com/pricing) currently includes $30 monthly credit. Listed GPU rates are approximately $0.80/hour for L4 and $0.59/hour for T4, before CPU and memory charges. Verify account-specific rates and remaining credit before each run.

Set the workspace [Spend limit](https://modal.com/docs/guide/budgets) to $0. The default can allow paid usage after credits. A local elapsed-time guard alone does not enforce a free-only workspace.

Use one GPU container, explicit step and elapsed-time caps, no automatic retry loop, and checkpoints on a [Modal Volume](https://modal.com/docs/examples/long-training). Run a short real-graph forward/backward benchmark before sizing the training run. Preserve model, optimizer, seed/RNG state, data hash, observation/action versions, and measured evaluation results.

## Scientific limits

This is a poker network constrained by measured fruit-fly connectivity. It is a computational rate-model approximation, not a reconstruction of complete biological dynamics, and does not show that a living fly learns poker.

## Measured full-graph benchmark

The first Modal GPU dispatch was rejected because GPU access required a payment method. The experiment switched to CPU compute rather than adding paid access.

A full-graph CPU benchmark completed on 2026-10-05. With eight observations per batch and two PyTorch threads, three forward/backward optimizer steps took 2.00, 2.23, and 2.34 seconds. Sensory adapter weights changed. The graph was the real checksum-verified release, not a synthetic test graph. See [the measured report](../experiments/real-graph-benchmark.json).

At the checked rates, four physical CPU cores and 8 GiB RAM cost approximately $0.2532/hour. A 20-minute worker ceiling is approximately $0.0844 before image-build overhead. These are resource estimates, not a final invoice. Account limits and credits must still be checked before launching.
