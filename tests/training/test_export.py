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
