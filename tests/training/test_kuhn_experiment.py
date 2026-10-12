from training.kuhn_experiment import run


def test_kuhn_experiment_reproducible_actual_average_strategy():
    first = run(seed=17, hands=64, seconds=30, hidden=8, batch=8)
    second = run(seed=17, hands=64, seconds=30, hidden=8, batch=8)
    assert first['before'] == second['before']
    assert first['after'] == second['after']
    assert first['completed_hands'] == 64
    assert first['average_strategy'] == second['average_strategy']
    assert len(first['average_strategy']) == 12
    assert first['measurement'] == 'exact best response to learned average strategy'
    assert first['improved'] == (first['after']['exploitability'] < first['before']['exploitability'])
    import json
    assert json.loads(json.dumps(first))['completed_hands'] == 64
