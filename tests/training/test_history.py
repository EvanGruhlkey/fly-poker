from training.environment import Holdem
from training.features import encode

def test_public_history_records_payments_relative_to_observer():
    env = Holdem(17)
    env.step(1)
    obs = env.observe()
    event = obs.history[-1]
    assert (event.actor, event.street, event.kind, event.paid) == (False, 0, 'call', 2)
    values = encode(obs)
    assert len(values) == 267
    assert values[118] == 0 and values[119] == 1
    assert values[120] == 1 and values[126] == 1
    assert values[128] == 2 / 800
    assert not hasattr(event, 'hole')
    env.step(1)
    assert env.observe().history[-1].kind == 'check'
    assert env.observe().history[-1].paid == 0


def test_context_features_preserve_prefix_and_expose_exact_actions():
    from dataclasses import replace
    import numpy as np
    from training.features import OBSERVATION_VERSION
    obs = Holdem(17).observe()
    values = encode(obs)
    assert OBSERVATION_VERSION == 'cards-context-v3'
    assert np.array_equal(values[:250], encode(obs, version='cards-history-v2'))
    effective = min(obs.stack, obs.opponent_stack)
    assert np.allclose(values[250:253], [effective / 400, effective / (effective + obs.pot), obs.to_call / (obs.pot + obs.to_call)])
    assert np.allclose(values[253:260], [action[1] / 400 if action else 0 for action in obs.actions])
    assert np.array_equal(values[260:], obs.mask)
    assert np.isfinite(encode(replace(obs, pot=0, stack=0, opponent_stack=0, to_call=0))).all()
