# Training a stronger fly

The user approved implementing the research recommendation on 2026-10-07.
The objective is a browser fly that reliably beats the equity baseline and
improves over the released policy across stack sizes and opponent styles.
Measured learning, rather than a successful optimizer run, decides promotion.

## Selected design

Train a small conventional NFSP teacher, then distill its average strategy
into the history-aware connectome. Keep the seven-action abstraction and
private-information boundary shared with the game. The browser still runs
the full fly graph, without an observation-to-policy bypass.

Two design explorations compare standalone teacher and in-graph NFSP.
Prefer a standalone teacher because replay, strategy averaging, and many
environment decisions do not require repeated full-graph backpropagation.
Retain in-graph self-play as a measured optional refinement.

NFSP maintains separate action-value and average-policy networks for each
seat, replay transitions, and a reservoir of best-response actions. Select
best-response versus average mode once per player per hand. A transition
connects a player's decision to that player's next decision or settlement.
Never bootstrap on the other player's private observation. Mask legal
actions in exploration, policy sampling, and action-value targets.

Validate game-theoretic behavior in Kuhn poker with exact best responses and
known game value. Holdem uses the existing PokerKit environment and 250
features. Its last-12-action observation is an approximation; do not claim
equilibrium convergence in the full game.

Distillation examples contain features, legal masks, and teacher action
probabilities. Collect from both teacher and student trajectories, balance
streets where possible, and train cross entropy over legal actions. Keep
training, validation, and final-test seeds disjoint. Record teacher identity
and schema in artifacts. Resume includes optimizer, replay/reservoir, progress,
all random states, and opponent archive where applicable.

Refinement uses historical opponents plus random, passive, equity, and teacher
opponents. Keep optional imitation regularization to limit forgetting.
Report imitation-only and refined policies separately.

Evaluation uses identical deals with seats swapped, paired confidence
intervals, opponent panels, 100-BB and unequal effective stacks, sampled
policies, and distinct validation/final-test domains. Several training seeds
test reproducibility. Exact exploitability is reported only for small games.
Promotion requires positive lower confidence bounds against equity and
paired improvement over the published starting policy, plus schema/export
parity and no illegal actions. Report incomplete evaluations explicitly.

All experiments have hand/update and wall-time caps, atomic checkpoints,
measured throughput, and machine-readable reports. Initial scale checkpoints
are 100,000 then 1,000,000 teacher hands, contingent on measured improvement.
Do not change account spend limits or add payment methods. Use local compute
or existing verified credit limits. Preserve the released model unless a
candidate passes promotion. Keep commits at or below 250 changed lines.

## Review focus

- Opponent turns must not leak private observations into replay targets.
- Replay and reservoir sampling must resume deterministically.
- Illegal action probabilities must remain zero through distillation.
- Teacher labels must cover positions reached by the student.
- Truncated runs must not receive complete or promoted status.
- Browser stacks carry across hands, so evaluate unequal stacks and matches.

## Throughput checkpoint

- Blocking first steps: baseline tests, architecture, shared interfaces.
- Independent workstreams: teacher implementation and evaluation extension.
- Shared mutable state: separate writer worktrees; serialize integration.
- Smallest safe decomposition: one owner for coupled teacher/distillation
  interfaces, one evaluation owner, independent final review.
