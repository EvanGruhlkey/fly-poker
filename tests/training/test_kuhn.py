import numpy as np
import pytest
from training.kuhn import Kuhn, equilibrium, assess


def test_known_equilibrium_has_game_value_and_zero_exploitability():
    result = assess(equilibrium)
    assert result['value_player_zero'] == pytest.approx(-1 / 18)
    assert result['exploitability'] == pytest.approx(0, abs=1e-12)


def test_passive_strategy_can_be_exploited_and_terminal_payoffs_conserve():
    assert assess(lambda card, history: np.array([1., 0.]))['exploitability'] > .1
    game = Kuhn(3)
    game.step(1)
    game.step(0)
    assert game.terminal
    assert (game.reward(0), game.reward(1)) == (1., -1.)
