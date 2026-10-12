# Poker strength audit and experiment

Measured on 2026-10-11. The Holdem strength objective remains unmet. The experimental v4 checkpoint was not promoted.

## Authorized follow-up round

The user reported $28 of Modal credits and authorized more compute. This comparison round uses at most two concurrent bounded L4 jobs with an estimated $3 aggregate ceiling. Current configured rates imply approximately $0.9586/hour for L4 plus two CPU cores and 8 GiB, or $1.0532/hour with four CPU cores. These are compute estimates, not an account billing statement.

The old soft equity labels assumed all bets were called at unchanged equity. On a 64-state diagnostic they assigned 46.1% mean probability to all-in. Baseline imitation now uses 98% of the original equity heuristic's selected action with 2% legal smoothing. Its value remains an approximate net showdown return; this does not make the heuristic a stronger teacher.

Round two compares 32 imitation steps followed by PPO against 512 cached imitation steps followed by PPO, both migrated from v2 with seed 31. Training and validation caches contain 4,096 and 512 states, use 64 equity samples, and have distinct heldout deal domains. Configurations are saved in `experiments/round2-ppo.json` and `experiments/round2-imitation.json`.

The first five-minute PPO continuation completed 64 updates and 1,024 hands. Mean rollout time was 2.167 seconds per update and mean gradient time was 0.360 seconds. The saved checkpoint preserves finite parameters and fixed connection signs. Its compute estimate is $0.082, excluding startup and storage. These measurements establish throughput, not improved strength.

A compact 267→64→64 training-only policy completed 256 imitation steps and 62 PPO updates over 7,936 hands in 60 local CPU seconds. It must pass a separate equity benchmark before supplying stronger supervision. From-scratch real, shuffled, zero-recurrence, and MLP controls use identical cached batches and bounded resources; equal steps and wall limits do not imply equal FLOPs.

The compact continuation reached 512 updates and 65,536 hands in another 206 CPU seconds. Its disjoint 1,536-hand validation still failed: equal, short, and medium stack means were +0.133, −0.498, and −0.365 BB/hand. No paired improvement interval excluded zero. This policy supplies no fly targets.

The Attack the Premise skill prompted a rerunnable coverage census (`experiments/dataset_census.py`). The shared premise was that the sampled imitation states adequately covered pressure decisions. The original cache had 3,078 of 4,096 states requiring no call, and only 394 of 3,072 postflop states facing a call. Fold labels were rare because pressure was rare: 62 of 131 states facing at least 25 BB correctly had fold as the teacher's preferred action. Increasing updates alone did not fix the sampling assignment.

The optional balanced-pressure sampler rotates street and pot-odds strata on fully legal trajectories, mixes passive, random and raising behavior, and generates 25% of hands with unequal initial stacks. It preserves distinct train/validation deal domains. It changes state exposure, not the labeler. Natural-frequency tournaments remain separate. The new 504-state validation cache has exactly 168 states in each pressure bin, 336 facing a call and 89 facing at least 25 BB. This is evidence of better coverage, not stronger poker play.

Heldout scoring also reproduced a numerical failure: weighted certain-win samples yielded equity 1.0000000000000002, making the uncertainty square root invalid. Weighted probabilities are now bounded after reduction. The failing regression was reproduced before the fix, then passed. The initial interrupted scoring run provides no strength result; its compute remains counted in the budget ledger.

The completed natural-cache imitation tournament used 64 paired deals per opponent and stack profile. It lost to equity at all three profiles: −9.066, −4.230, and −5.406 BB/hand, with each 95% interval below zero. Direct gains against v2 were −2.844, −0.789, and −0.734 BB/hand; those intervals included zero. Its heldout cross-entropy improved from v2's 2.500 to 0.798, but fold recall was zero on ten fold-target states. Lower imitation loss did not establish strength.

The detailed L4 profile measured approximately 12 ms forward, 24 ms backward and 1.2 ms optimizer time per batch of 16 after startup; input creation and transfer took 0.56 ms. The first iteration additionally incurred kernel setup. Frequent remote checkpoint commits dominated the 305-second imitation job: all 512 update computations totaled only 16.53 seconds. New jobs retain frequent local atomic checkpoints, throttle remote synchronization to 60 seconds, and force synchronization at boundaries. A worker crash can lose unsynchronized work; a normal budget stop restores and synchronizes a completed checkpoint. Tests preserve exact resume and rollback behavior. Adapter-only and completed-imitation snapshots now permit direct stage comparisons rather than retaining only the PPO result.

Independent review found missing sparse-kernel and checkpoint-migration dependencies in the evaluation resume identity. New reports pin those sources, graph-loading code, and runtime versions as well. Older pilot reports remain historical evidence and cannot silently resume across that provenance change.

## Findings and fixes

| Finding | Evidence and change |
| --- | --- |
| The teacher is weak and its response model ignores the board. | The previous response strength used only preflop ranks. Sampled opponents now estimate equity from their hole cards and the public board against independent unknown hands and runouts. |
| Large bets benefit from optimistic sampled fold equity. | Added a sample-size-dependent uncertainty penalty. The resulting teacher still fails its validation gate. |
| Imitation and reinforcement values have different accounting. | Both now target expected net whole-hand chips divided by 800, equivalent to terminal BB divided by 200. Imitation subtracts all earlier investment, including blinds. |
| Dividing rewards by 200 weakens unnormalized policy updates. | Retained this scale for the compatible tanh value head. PPO normalizes fixed behavior advantages, removing the global scale from policy gradients. |
| The original actor-critic has no clipping or behavior records. | Trajectories now store sampled-action log probabilities and values. PPO uses terminal Monte Carlo returns, clipped updates, bounded entropy, gradient clipping, and KL stopping. |
| The only fly opponent refreshes every iteration. | Added persistent historical snapshots, refreshed at configurable intervals. Random, passive, equity, experimental teacher, current fly, and historical fly opponents form the league. |
| Observations omit exact sizes and legality. | Appended effective stack, bounded SPR, pot odds, seven exact action amounts, and seven mask bits. The existing 12-event public betting history remains. |
| Seat balancing is implicit and dealer is fixed in rollout. | Training now randomizes dealer as well as learner seat. Evaluation swaps seats and matches decks. |
| Stack sampling can mismatch browser play. | Retained 100 BB for imitation and early PPO. Later PPO uses 75% equal stacks and 25% unequal stacks. Evaluation separates stack profiles. |
| Cached training and validation can share deals. | Review caught Python's identical integer seeding for positive and negative values. Dataset v2 uses distinct seed magnitudes and tests actual decks. Prior dataset schemas are rejected. |
| Time limits can save half an update. | Interrupted training reloads the last completed checkpoint, including model, optimizer, RNG, and progress. Incomplete work is discarded. |
| One AA-versus-72 probe could publish a model. | The preexisting poker-fit path now rejects production publication. A hand-strength probe cannot establish poker strength. |

The policy path remains observation, sensory projection, eight recurrent FlyWire steps, motor neurons, policy and value heads. No direct observation bypass was added. All graph edges and excitatory or inhibitory signs remain fixed.

The Model the Domain principle guided the versioned observation, trajectory, dataset, and frozen-opponent records. Prove It Works required actual L4 execution and held-out measurements rather than treating lower loss as strength. Independent agent review used the same inherited model, so model diversity was reduced. Review found the seed-sign leakage, partial-update checkpoint risk, and insufficient dataset provenance. The fixes have regression tests.

Current verification passed 130 Python tests, 42 browser tests, the production build, and `git diff --check`. Tests cover legacy and current browser parity, uninterrupted versus resumed updates, interrupted PPO rollback, frozen historical policies, and replacement-dataset rejection. The preexisting Vite bundle-size warning remains. Changes remain uncommitted.

## Observation and checkpoint contract

`cards-context-v3` has 267 features. Columns 0 through 249 retain `cards-history-v2`. Columns 250 through 252 contain effective remaining stack divided by 400, effective stack divided by effective stack plus pot, and call divided by pot plus call. Columns 253 through 259 contain exact abstract action amounts divided by 400. A raise amount is a raise-to target; a call amount is incremental payment. Columns 260 through 266 contain legal-action bits.

The model receives only own cards, public board, and public table information. Monte Carlo equities, sampled opponent cards, and future runouts are teacher-only data. Public history records actor relative to the observer, street, action kind, and incremental payment.

New checkpoints and exports use v4. Explicit v2 and v3 migrations copy all compatible parameters and zero new sensory columns. Browser inference still accepts released 118- and 250-feature exports. Python and Node tests compare full encodings on legal hands and forward outputs for all supported widths. The production manifest already had uncommitted changes when this task began; this run did not edit or publish it.

## Teacher limitations and gate

The candidate samples unknown opponent hands and runouts without reading the environment's hidden state. It weights a heuristic preflop range by public opponent aggression. For each legal bet it estimates calling probability from independently sampled opponent equity and pot odds. Payoffs account for matched chips, fold equity, and unmatched refunds. It shares showdown samples across action candidates.

This is a one-decision showdown approximation, not a complete betting-tree rollout. It does not model future strategic betting or fit ranges to observed opponents. Its eight-sample opponent response estimate remains noisy. The uncertainty penalty is an engineering approximation rather than a poker theorem. Improving this response model is the next research priority.

The teacher must beat the equity heuristic with positive lower confidence bounds across all three stack profiles, using at least 128 completed validation pairs. Rollout supervision requires a matching teacher source hash and Monte Carlo sample count. Cached labels also carry a checksum and teacher hash. Resume pins dataset checksums rather than trusting paths.

The first exploratory benchmark returned -4.84 BB per hand, CI [-14.90, 5.73], over 32 pairs. After the uncertainty penalty, another exploratory benchmark returned +0.78, CI [-4.95, 6.56], over 64 pairs. These use different seeds and do not isolate the penalty's causal effect.

The reproducible validation benchmark used seed 9417 and 32 samples. It exhausted 180 seconds. Equal stacks completed 128 pairs and returned +0.758 BB per hand, CI [-3.327, 4.934]. The 20-BB profile completed only 13 pairs and the 50-BB profile completed none. The gate rejected the teacher. See [the measured report](../experiments/teacher-validation.json). The GPU pipeline smoke therefore used the existing equity teacher, not an assumed stronger teacher.

## Staged optimization and resume

Stage A freezes synaptic magnitudes, neuron biases, and leaks. Only sensory and motor adapters learn. Stage B unfreezes those allowed graph parameters, applies mean-square drift regularization, and bounds parameter changes to 0.1 around initial values. The activation saturation and leak parameterization remain unchanged. PPO uses the same anchors and bounds.

`adapter_steps` counts completed imitation steps plus PPO updates. Step totals are bounded at 100,000, batch size at 16, PPO epochs at eight, and resident historical snapshots at four. Modal retains one L4 container, no retries, a 1,200-second worker timeout, and a configurable training budget capped at 900 seconds. Larger experiments require explicit bounded continuations. These limits were not removed.

Checkpoints save model, optimizer, progress, NumPy RNG, CPU Torch RNG, CUDA RNG, graph identity, observation and action versions, configuration, and dataset fingerprints. Interrupted work rolls back to the latest periodic checkpoint. Tests verify continuous versus resumed parameter equality and interruption after a real PPO optimizer update.

## Measured L4 pilot

The checksum-verified full graph contains 134,209 neurons and 2,700,513 connections. One batch-four forward/backward optimizer step on L4 took 1.109 seconds and peaked at 286,557,696 allocated CUDA bytes. This is a smoke measurement, not steady-state profiling. Sparse backward skips edge-gradient calculation while graph weights are frozen. AMP was not enabled.

The bounded pilot migrated published v2, trained eight imitation steps and four PPO updates over 16 hands, and completed in 24.337 seconds. It saved frozen snapshots and a resumable checkpoint in `fly-poker-artifacts` at `/runs/brain-v4-pilot-seed-23/latest.pt`. All parameters were finite and fixed signs remained intact. See [the pilot report](../experiments/brain-v4-pilot.json) and [its configuration](../experiments/pilot-v4.json).

The four-pair in-training evaluation was inconclusive for relative improvement and negative against equity. A separate held-out debugging evaluation used eight pairs, the published v2, a previous v3 checkpoint, and seven opponent policies. It exhausted its 180-second budget before completing every stack profile. At 100 BB the candidate returned -23.969 BB per hand against equity, CI [-37.846, -9.844]. Its paired gain over v2 against equity was +1.688, CI [-15.063, 15.000]. This is insufficient evidence of improvement. See [the full debugging report](../experiments/v4-heldout-debug.json).

The promotion gate rejected the candidate. Final comparisons require at least 1,000 paired decks per profile, the final test seed domain, the complete panel, positive confidence bounds, and independently verified artifact parity. The debugging evaluator does not attest parity for a particular candidate export.

Modal's authenticated rates were $0.80 per L4 hour, $0.0473 per CPU core hour, and $0.008 per GiB hour. The requested two CPU cores and 8 GiB give $0.9586 per container hour before other charges. The pilot's measured training and evaluation duration implies roughly $0.0065 at those rates, excluding startup and billing differences. This is an estimate, not an invoice.

The initial billing CLI reported $1.11271545 in compute usage, $1.11 in applied credits, and zero billed cost. It did not expose remaining credit or the hard spend limit. The user subsequently reported $28 in credits and authorized bounded followup experiments. Round two has a $3 estimated compute ceiling; its ledger uses measured durations where available and caps otherwise. This is separate from actual account billing.

## Separate imperfect-information experiment

[NFSP](https://arxiv.org/abs/1603.01121) combines reinforcement-learning best responses with supervised average-strategy learning. [Deep CFR](https://arxiv.org/abs/1811.00164) approximates counterfactual regrets through neural networks and sampled game traversals. The [OpenSpiel example](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/examples/deep_cfr_pytorch.py) demonstrates its dedicated solver workflow. [ReBeL](https://arxiv.org/abs/2007.13544) combines learning and search over public belief states. None is a drop-in replacement for this actor-critic.

The existing compact NFSP implementation ran 2,000 Kuhn hands in 9.78 seconds with seed 17. Exact average-strategy exploitability decreased from 0.4334 to 0.1927. See [the measured Kuhn report](../experiments/kuhn-nfsp-pilot.json). One seed in Kuhn does not validate Holdem NFSP or CFR targets. A full ReBeL implementation was not attempted.

## Round-two diagnosis

Natural-state sampling put 75% of examples in states with no call required. The balanced-pressure cache now explicitly covers four streets and three pot-odds strata, with unequal stacks in part of the trajectory curriculum. On 504 heldout states, pressure imitation recovered 46% of teacher fold choices, compared with 0% for v2. Its raise-category recall remained only 17.6%, so this is partial behavioral learning rather than a strong strategy.

The original soft labels also assumed every bet was called at unchanged equity, assigning 46% mean probability to all-in in a 64-state diagnostic. The baseline imitation labels now reproduce the existing equity heuristic with legal smoothing and consistent net-return values. They are a diagnostic baseline; the stronger information-set teacher and compact policies failed their independent strength gates.

Checkpoint I/O dominated cached training. Keeping frequent atomic local saves while synchronizing the remote volume at most once per minute reduced the 1,024-step pressure imitation run to 63.8 seconds, including 19.8 seconds of local saving and 5.4 seconds of remote commits. Adapter-only and completed imitation snapshots are retained for stage comparisons.

Pressure PPO completed 172 updates over 2,752 hands. Small per-update KL values did not prevent accumulated policy drift: the final model's fold recall fell from 46% to 0% and raising dominated its heldout action mass. The matched tournament comparison is recorded separately. Finite parameters, preserved edge signs, and lower training loss are necessary checks, not evidence of poker improvement. Independent read-only review found no additional correctness issues in the checkpoint throttle or stage snapshots.

## Remaining research

The stronger teacher is not validated. High-volume rollout supervision remains gated. Detailed L4 profiling and one-seed wall-budget retraining of shuffled-graph, zero-edge, and MLP controls are complete; their exploratory confidence intervals do not establish a biological topology advantage. Multiple training seeds and thousands-of-hands final evaluations remain open. No new production brain was exported. See [round-two results](round2-results.md) for the bounded followup comparisons.
