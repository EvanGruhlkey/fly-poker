from dataclasses import replace
import numpy as np
from training.environment import Holdem, PublicAction
from training.teacher import action_rewards, action_targets, decision_values, poker_targets, range_weight, equity

def test_teacher_values_incremental_cost_and_balances_legal_targets():
    obs=Holdem(17).observe()
    weak,_=action_targets(obs,.12);strong,_=action_targets(obs,.95)
    assert weak[0]>weak[5] and strong[5]>strong[0]
    assert np.isclose(weak.sum(),1)
    assert np.all(weak[np.logical_not(obs.mask)]==0)
    short=replace(obs,stack=1,opponent_stack=795,actions=(('fold',0),('check_call',1),None,None,None,None,None),to_call=1)
    probs,value=action_targets(short,.8)
    assert probs[1]>probs[0] and np.isfinite(value)

def test_public_aggression_weights_generated_strong_ranges():
    obs=Holdem(17).observe()
    assert range_weight(obs,('As','Ad'))==range_weight(obs,('7c','2d'))==1
    raised=replace(obs,history=(PublicAction(False,0,'raise',80),))
    assert range_weight(raised,('As','Ad'))>range_weight(raised,('7c','2d'))
    assert 0<=equity(raised,np.random.default_rng(4),samples=64)<=1


def test_short_all_in_teacher_excludes_uncalled_opponent_refund():
    obs=Holdem(17).observe()
    short=replace(obs,pot=100,committed=0,opponent_committed=100,stack=10,opponent_stack=690,
        actions=(('fold',0),('check_call',10),None,None,None,None,None),to_call=10)
    probabilities,value=action_targets(short,.5)
    assert np.allclose(probabilities[:2],[.5,.5])
    assert value==0
    prior_street=replace(short,pot=120)
    probabilities,value=action_targets(prior_street,.5)
    assert probabilities[1]>probabilities[0] and value>0

def test_fold_equity_rewards_a_bet_only_against_hands_the_bet_beats():
    obs=Holdem(17).observe()
    raises=[index for index,action in enumerate(obs.actions) if action and action[0]=='raise']
    check=next(index for index,action in enumerate(obs.actions) if action and action[0]=='check_call')
    everyone_folds=[(1.0,0.0,0.0)]*4
    everyone_calls=[(1.0,0.0,1.0)]*4
    folded,_=decision_values(obs,everyone_folds)
    called,_=decision_values(obs,everyone_calls)
    assert folded[raises[0]]>folded[check]
    assert called[check]>called[raises[0]]
    won=[(1.0,1.0,0.0),(1.0,1.0,1.0)]
    lost=[(1.0,0.0,0.0),(1.0,0.0,1.0)]
    strong,_=decision_values(obs,won)
    weak,_=decision_values(obs,lost)
    assert strong[raises[-1]]>weak[raises[-1]]
    assert weak[0]+weak[check]>strong[0]+strong[check]

def test_called_shove_pays_worse_than_a_check_when_the_fly_is_behind():
    obs=Holdem(17).observe()
    raises=[index for index,action in enumerate(obs.actions) if action and action[0]=='raise']
    check=next(index for index,action in enumerate(obs.actions) if action and action[0]=='check_call')
    rewards=action_rewards(obs,[(1.0,0.0,1.0)])
    assert rewards[check]>rewards[raises[-1]]

def test_poker_targets_stay_on_legal_actions():
    obs=Holdem(17).observe()
    probabilities,value=poker_targets(obs,np.random.default_rng(1),samples=4)
    assert np.isclose(probabilities.sum(),1)
    assert np.all(probabilities[np.logical_not(obs.mask)]==0)
    assert np.isfinite(value)


def test_weighted_certain_wins_remain_finite_despite_reduction_rounding():
    obs = Holdem(17).observe()
    rng = np.random.default_rng(1)
    with np.errstate(invalid='raise'):
        for _ in range(100):
            samples = [(float(weight), 1., 1.) for weight in rng.random(16)]
            probabilities, value = decision_values(obs, samples)
            assert np.isfinite(probabilities).all() and np.isfinite(value)
            assert np.isclose(probabilities.sum(), 1)
