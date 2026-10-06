import numpy as np
from .environment import Observation

OBSERVATION_VERSION = 'cards-public-v1'
FEATURE_COUNT = 118


def card_index(card: str) -> int:
    return '23456789TJQKA'.index(card[0]) * 4 + 'cdhs'.index(card[1])


def encode(obs: Observation) -> np.ndarray:
    features = np.zeros(FEATURE_COUNT, dtype=np.float32)
    for card in obs.hole:
        features[card_index(card)] = 1
    for card in obs.board:
        features[52 + card_index(card)] = 1
    features[104 + obs.street] = 1
    raises = [action[1] for action in obs.actions if action is not None and action[0] == 'raise']
    features[108:] = [float(obs.dealer), obs.pot / 800, obs.to_call / 400,
        obs.stack / 400, obs.opponent_stack / 400, obs.committed / 400,
        obs.opponent_committed / 400, len(obs.board) / 5,
        min(raises, default=0) / 400, max(raises, default=0) / 400]
    return features
