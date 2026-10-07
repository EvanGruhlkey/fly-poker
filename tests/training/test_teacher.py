from dataclasses import replace
import numpy as np
from training.environment import Holdem, PublicAction
from training.teacher import action_targets, range_weight, equity

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
