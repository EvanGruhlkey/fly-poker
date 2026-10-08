import numpy as np
from .features import encode


def observe(game):
    if hasattr(game, 'features'):
        return game.features(), game.mask()
    observation = game.observe()
    return encode(observation), np.array(observation.mask, dtype=bool)


def collect_hand(game, agents, rng, *, eta, epsilon, reward_scale=1.):
    modes = rng.random(2) < eta
    pending = [None, None]
    decisions = 0
    while not game.terminal:
        seat = game.actor
        features, mask = observe(game)
        if pending[seat] is not None:
            previous, legal, action = pending[seat]
            agents[seat].transition(previous, legal, action, 0., features, mask, False, rng)
        action = agents[seat].act(features, mask, bool(modes[seat]), epsilon, rng)
        pending[seat] = (features, mask, action)
        game.step(action)
        decisions += 1
        if decisions > 400:
            raise RuntimeError('NFSP hand exceeded finite betting bound')
    for seat in (0, 1):
        if pending[seat] is not None:
            features, mask, action = pending[seat]
            agents[seat].transition(features, mask, action, game.reward(seat) / reward_scale,
                                    np.zeros_like(features), np.zeros_like(mask), True, rng)
    return {'decisions': decisions, 'best_response_modes': modes.tolist()}
