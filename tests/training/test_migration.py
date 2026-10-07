import numpy as np
import torch
import pytest
from training.checkpoint import migrate_v2, load_checkpoint
from training.data import tiny_graph
from training.features import FEATURE_COUNT
from training.model import BrainPolicy, ModelConfig

def test_explicit_migration_preserves_logits_and_zeroes_history(tmp_path):
    old = BrainPolicy(tiny_graph(), ModelConfig(observation_features=118))
    payload={'format':'fly-poker-checkpoint-v2','metadata':{**old.metadata(),'observation_version':'cards-public-v1','action_version':'holdem-seven-v1'},'model_state':old.state_dict()}
    source=tmp_path/'old.pt';torch.save(payload,source)
    new=BrainPolicy(tiny_graph());info=migrate_v2(source,new)
    assert info['source_sha256'] and info['source_observation_version']=='cards-public-v1'
    assert torch.count_nonzero(new.sensory.weight[:,118:])==0
    features=torch.rand(3,FEATURE_COUNT);mask=torch.ones(3,7,dtype=torch.bool)
    old_logits,old_value=old(features[:,:118],mask);new_logits,new_value=new(features,mask)
    assert torch.allclose(old_logits,new_logits,atol=1e-6)
    assert torch.allclose(old_value,new_value,atol=1e-6)
    with pytest.raises(ValueError):load_checkpoint(source,new,torch.optim.Adam(new.parameters()),np.random.default_rng())
    payload['model_state']['signs'][0]*=-1;torch.save(payload,source)
    with pytest.raises(ValueError):migrate_v2(source,new)
