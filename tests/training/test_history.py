from training.environment import Holdem
from training.features import encode

def test_public_history_records_payments_relative_to_observer():
    env = Holdem(17)
    env.step(1)
    obs = env.observe()
    event = obs.history[-1]
    assert (event.actor, event.street, event.kind, event.paid) == (False, 0, 'call', 2)
    values = encode(obs)
    assert len(values) == 250
    assert values[118] == 0 and values[119] == 1
    assert values[120] == 1 and values[126] == 1
    assert values[128] == 2 / 800
    assert not hasattr(event, 'hole')
    env.step(1)
    assert env.observe().history[-1].kind == 'check'
    assert env.observe().history[-1].paid == 0
