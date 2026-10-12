import numpy as np
from training.environment import Holdem
from training.teacher import heuristic, heuristic_targets


def test_heuristic_imitation_does_not_turn_modest_equity_into_shove_labels():
    obs = Holdem(7).observe()
    probabilities, value = heuristic_targets(obs, .55)
    assert probabilities.argmax() == heuristic(obs, .55) == 1
    assert probabilities[1] > .98
    assert probabilities[6] < .01
    assert np.isclose(probabilities.sum(), 1)
    assert np.all(probabilities[~np.asarray(obs.mask)] == 0)
    assert np.isfinite(value)
