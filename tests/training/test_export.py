import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import torch
from training.data import tiny_graph
from training.export import export_model
from training.features import FEATURE_COUNT
from training.model import BrainPolicy


def test_browser_export_matches_python_sparse_model(tmp_path):
    torch.manual_seed(7)
    model = BrainPolicy(tiny_graph())
    features = torch.rand(3, FEATURE_COUNT)
    masks = torch.tensor([[True]*7, [False, True, False, True, False, False, False], [True]*7])
    logits, values = model(features, masks)
    export_model(model, tmp_path)
    (tmp_path / 'inputs.json').write_text(json.dumps({'features': features.tolist(), 'masks': masks.tolist()}))
    runtime = (Path(__file__).parents[2] / 'web/src/brain/runtime.mjs').as_uri()
    script = f"""import fs from 'node:fs'; import {{parseModel,forward}} from '{runtime}';
const root=process.argv[1]; const manifest=JSON.parse(fs.readFileSync(root+'/manifest.json'));
const b=fs.readFileSync(root+'/model.bin');const model=parseModel(manifest,b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));
const inputs=JSON.parse(fs.readFileSync(root+'/inputs.json'));
console.log(JSON.stringify(inputs.features.map((x,i)=>forward(model,x,inputs.masks[i]))));"""
    output = subprocess.check_output([shutil.which('node'), '--input-type=module', '-e', script, str(tmp_path)], text=True)
    results = json.loads(output)
    for index, result in enumerate(results):
        valid = masks[index].numpy()
        assert np.allclose(np.array(result['logits'])[valid].astype(float), logits[index].detach().numpy()[valid], atol=3e-5)
        assert abs(result['value'] - values[index].item()) < 3e-5

def test_browser_public_features_and_seven_wagers_match_pokerkit(tmp_path):
    from training.environment import Holdem
    from training.features import encode
    rng = np.random.default_rng(91)
    observations = []
    expected = []
    for seed in range(20):
        buyin = int(rng.integers(4, 797))
        env = Holdem(seed, dealer=seed%2, stacks=(buyin, 800-buyin))
        while not env.terminal:
            obs = env.observe()
            state = env._state
            raises = [action[1] for action in obs.actions if action and action[0]=='raise']
            legal = {'call': obs.to_call, 'canFold': bool(obs.actions[0]), 'canCheck': obs.to_call==0,
                'raise': {'kind':'raise','minTo':state.min_completion_betting_or_raising_to_amount,
                    'maxTo':state.max_completion_betting_or_raising_to_amount} if raises else {'kind':'none'}}
            observations.append({'holeCards':obs.hole,'board':obs.board,'street':obs.street,'dealer':obs.dealer,
                'pot':obs.pot,'stack':obs.stack,'opponentStack':obs.opponent_stack,'committed':obs.committed,
                'opponentCommitted':obs.opponent_committed,'legal':legal})
            expected.append((encode(obs), obs.actions))
            env.step(int(rng.choice(np.flatnonzero(obs.mask))))
    (tmp_path/'observations.json').write_text(json.dumps(observations))
    domain = (Path(__file__).parents[2]/'web/src/brain/domain.mjs').as_uri()
    script = f"""import fs from 'node:fs';import {{abstractActions,encodeObservation}} from '{domain}';
const observations=JSON.parse(fs.readFileSync(process.argv[1]));
console.log(JSON.stringify(observations.map(x=>({{features:Array.from(encodeObservation(x)),actions:abstractActions(x)}}))));"""
    actual = json.loads(subprocess.check_output([shutil.which('node'),'--input-type=module','-e',script,str(tmp_path/'observations.json')],text=True))
    for result,(features,actions) in zip(actual,expected,strict=True):
        assert np.allclose(result['features'],features,atol=1e-7)
        assert [action is not None for action in actions] == [action is not None for action in result['actions']]
        for python_action,js_action in zip(actions,result['actions']):
            if python_action and python_action[0]=='raise':
                assert python_action[1] == js_action['to']
