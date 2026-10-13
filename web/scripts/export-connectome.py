import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[2]
graph = np.load(root / 'data/flywire/full.npz')
positions = graph['position'][:, :2].astype(float)
valid = np.isfinite(positions).all(axis=1)
lo = positions[valid].min(axis=0)
span = positions[valid].max(axis=0) - lo
positions = (positions - lo) / span
classes = graph['super_class']
colors = {'optic': '#56c496', 'central': '#e9a63a', 'sensory': '#8be05a',
          'visual_projection': '#60d2dc', 'visual_centrifugal': '#78aaf0',
          'ascending': '#ffcb6b', 'descending': '#ff7846', 'motor': '#ff5046',
          'sensory_ascending': '#bef078', 'endocrine': '#f082c8'}
sources = graph['csr_indices']
ptr = graph['csr_indptr']
ids = set()
for i in range(1024):
    edge = i * len(sources) // 1024
    ids.add(int(sources[edge]))
    ids.add(int(np.searchsorted(ptr, edge, side='right') - 1))

def point(i):
    return [round(float(x), 5) for x in positions[i]] + [colors.get(str(classes[i]), '#969696')]

silhouette_ids = np.flatnonzero(valid)[::12]
data = {'neurons': {str(i): point(i) for i in sorted(ids) if valid[i]},
        'silhouette': [point(int(i)) for i in silhouette_ids]}
path = root / 'web/src/brain/anatomy.mjs'
path.write_text('export const anatomy=' + json.dumps(data, separators=(',', ':')) + ';\n', encoding='utf-8')
print(f'Exported {len(data["neurons"])} activity positions and {len(silhouette_ids)} silhouette neurons')
xyz = graph['position'].astype(float)
finite = np.isfinite(xyz).all(axis=1)
low, high = xyz[finite].min(axis=0), xyz[finite].max(axis=0)
xyz = np.nan_to_num((xyz - (low + high) / 2) / (high - low).max() * 2)
palette = dict(colors, optic='#528ed0', central='#e7b247', sensory='#50da91', motor='#f45b57', descending='#f45b57')
rgb = np.array([[int(palette.get(str(c), '#748ca1')[i:i+2], 16) / 255 for i in (1, 3, 5)] for c in classes])
out = root / 'web/public/connectome'
out.mkdir(exist_ok=True)
np.concatenate([xyz, rgb], axis=1).astype('<f4').tofile(out / 'anatomy.bin')
