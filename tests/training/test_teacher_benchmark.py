import json
import pytest

from training.teacher_benchmark import benchmark, validation_gate


def report(lower=.1, *, pairs=128, complete=True):
    entry = {'complete': complete, 'paired_samples': pairs, 'ci95': [lower, 3.]}
    return {'config': {'domain': 'validation', 'pairs': pairs}, 'complete': complete,
        'profiles': {name: {'candidate_vs_equity': entry.copy(), 'paired_gain_vs_equity': entry.copy()}
            for name in ('400_400', '80_720', '200_600')}}


def test_teacher_gate_requires_positive_completed_validation():
    assert validation_gate(report())['accepted']
    for candidate in (report(-.1), report(0), report(pairs=127), report(complete=False)):
        assert not validation_gate(candidate)['accepted']
    invalid = report()
    invalid['config']['domain'] = 'test'
    assert not validation_gate(invalid)['accepted']
    invalid = report()
    invalid['profiles']['80_720']['candidate_vs_equity']['ci95'] = [float('nan'), 2.]
    assert not validation_gate(invalid)['accepted']


def test_benchmark_records_incomplete_evidence_without_acceptance(tmp_path):
    output = tmp_path / 'teacher.json'
    result = benchmark(pairs=1, seed=42, samples=1, seconds=0, output=output)
    assert not result['complete']
    assert not result['validation_gate']['accepted']
    assert result['teacher_version']
    assert result['elapsed_seconds'] >= 0
    saved = json.loads(output.read_text())
    assert saved['config']['domain'] == 'validation'
    assert saved['config']['pairs'] == 1


@pytest.mark.parametrize('pairs', [0, 10001])
def test_benchmark_bounds_pair_count(pairs):
    with pytest.raises(ValueError):
        benchmark(pairs=pairs, seed=42, samples=1, seconds=0, output=None)


def test_benchmark_policy_sampling_is_independent_of_labels(monkeypatch):
    import numpy as np
    import training.teacher_benchmark as module
    from training.environment import Holdem

    def evaluate(candidate, starting, budget, config, *, opponents):
        observation = Holdem(17).observe()
        assert opponents['equity'] is starting
        assert np.array_equal(candidate(observation), candidate(observation))
        assert np.array_equal(starting(observation), starting(observation))
        return report(pairs=config.pairs)

    monkeypatch.setattr(module, 'tournament', evaluate)
    result = benchmark(pairs=128, seed=42, samples=2, seconds=30)
    assert result['candidate'] == 'poker_targets'
    assert result['baseline'] == 'original equity heuristic'
    assert result['validation_gate']['accepted']
