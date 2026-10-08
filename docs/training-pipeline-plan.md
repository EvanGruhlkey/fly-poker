# Training pipeline implementation plan

**Goal:** Train and measure an NFSP teacher and history-aware fly student.
**Architecture:** Small teacher networks generate mixed-strategy labels.
The existing connectome learns those labels and optionally refines against
an opponent archive. Evaluation decides whether to replace released weights.
**Tech stack:** Python 3.12, PyTorch, NumPy, PokerKit, existing browser runtime.
**Spec:** docs/training-pipeline-design.md

## Global constraints

- Preserve hidden-card isolation, seven actions, 250-feature history schema.
- No unbounded compute, automatic retries, or account spending changes.
- Every commit changes at most 250 lines; plain messages without trailers.
- Keep published weights until a candidate passes the measured gates.

## Tasks

1. NFSP replay/reservoir and policy networks in training/nfsp*.py.
   - [ ] Test seeded reservoir replacement, legal masking, and checkpoint
     round-trip before implementing them.
   - [ ] Implement next-own-decision transitions and episode-level mode.
   - [ ] Test uninterrupted versus resumed trajectories and terminal returns.
   - [ ] Verify and commit each independently testable unit.
2. Small-game validation in training/kuhn.py and tests/training/test_kuhn.py.
   - [ ] Test known equilibrium value -1/18 and zero exploitability.
   - [ ] Implement exact best-response enumeration and neural training adapter.
   - [ ] Run a measured NFSP learning experiment and save its report.
3. Teacher Holdem training and distillation in training/distill.py.
   - [ ] Test legal mixed probability labels and train/validation separation.
   - [ ] Add bounded resumable teacher CLI and atomic artifacts.
   - [ ] Test student-generated positions and falling held-out imitation loss.
   - [ ] Integrate full-graph training and measured export.
4. Evaluation in training/tournament.py and focused tests.
   - [ ] Test paired deals, stack/seat symmetry, domains, bootstrap intervals,
     complete versus truncated reports, and promotion rejection.
   - [ ] Add random, passive, equity, aggressive, teacher, and checkpoint rivals.
   - [ ] Support multiple training seeds and fixed final-test manifests.
5. Historical opponent refinement in training/train.py and checkpoint.py.
   - [ ] Test archive preservation and exact resume.
   - [ ] Add teacher regularization and retained snapshot sampling.
   - [ ] Remove pilot-only step/evaluation caps in favor of explicit budgets.
6. Execution, review, and delivery.
   - [ ] Run Python baseline and changed tests; run browser tests/build/parity.
   - [ ] Benchmark teacher and real graph; run bounded learning experiments.
   - [ ] Evaluate teacher, imitation student, and refined student separately.
   - [ ] Review algorithm correctness and experiment claims independently.
   - [ ] Document results, limitations, commands, and candidate promotion.
