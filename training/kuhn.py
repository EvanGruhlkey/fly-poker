from itertools import permutations, product
from random import Random
import numpy as np

HISTORIES = ('', 'c', 'b', 'cb')
TERMINALS = ('cc', 'bf', 'bc', 'cbf', 'cbc')
FEATURE_COUNT = 9


def payoff(cards, history):
    if history == 'bf':
        return 1.
    if history == 'cbf':
        return -1.
    return (1. if cards[0] > cards[1] else -1.) * (1 if history == 'cc' else 2)


def child(history, action):
    return history + (('f', 'c')[action] if history.endswith('b') else ('c', 'b')[action])


def equilibrium(card, history):
    aggressive = {'': (1 / 3, 0, 1), 'c': (1 / 3, 0, 1),
                  'b': (0, 1 / 3, 1), 'cb': (0, 2 / 3, 1)}[history][card]
    return np.array([1 - aggressive, aggressive])


def expected(policy, cards, history='', responder=None, choices=None):
    if history in TERMINALS:
        return payoff(cards, history)
    seat = len(history) % 2
    if seat == responder:
        probabilities = [0., 0.]
        probabilities[choices[(cards[seat], history)]] = 1.
    else:
        probabilities = policy(cards[seat], history)
    return sum(probabilities[action] * expected(policy, cards, child(history, action), responder, choices)
               for action in (0, 1))


def assess(policy):
    deals = tuple(permutations(range(3), 2))
    value = sum(expected(policy, cards) for cards in deals) / 6
    best = []
    for seat in (0, 1):
        infos = [(card, history) for card in range(3) for history in HISTORIES if len(history) % 2 == seat]
        utilities = []
        for actions in product((0, 1), repeat=len(infos)):
            choices = dict(zip(infos, actions, strict=True))
            utility = sum(expected(policy, cards, responder=seat, choices=choices) for cards in deals) / 6
            utilities.append(utility * (1 if seat == 0 else -1))
        best.append(max(utilities))
    return {'value_player_zero': value, 'best_response_values': best,
            'exploitability': sum(best) / 2, 'nash_conv': sum(best)}


class Kuhn:
    def __init__(self, seed):
        cards = [0, 1, 2]
        Random(seed).shuffle(cards)
        self.cards = cards[:2]
        self.history = ''

    @property
    def terminal(self):
        return self.history in TERMINALS

    @property
    def actor(self):
        return None if self.terminal else len(self.history) % 2

    def features(self):
        features = np.zeros(FEATURE_COUNT, dtype=np.float32)
        features[self.cards[self.actor]] = 1
        features[3 + self.actor] = 1
        features[5 + HISTORIES.index(self.history)] = 1
        return features

    def mask(self):
        return np.ones(2, dtype=bool)

    def step(self, action):
        if self.terminal or action not in (0, 1):
            raise ValueError('illegal Kuhn action')
        self.history = child(self.history, action)

    def reward(self, seat):
        if not self.terminal:
            raise ValueError('Kuhn reward requires settlement')
        return payoff(self.cards, self.history) * (1 if seat == 0 else -1)
