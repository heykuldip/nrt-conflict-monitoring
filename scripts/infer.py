"""
python -m scripts.infer CHECKPOINT BEFORE.tif AFTER.tif OUT.tif --normalisation FILE.yaml
    [--bands 3 2 1 4] [--alignment FILE.yaml]
--bands: 1-based red, green, blue and NIR bands (PlanetScope 4-band: 3 2 1 4). Scenes should share one grid.
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml
from torch.nn.functional import cosine_similarity

from src.data import alignment
from src.utils import load_obj

CHANNELS = Path(__file__).resolve().parent.parent / 'config' / 'channels' / 'rgb_nir.yaml'


def load_model(path):
    ckpt = torch.load(path, map_location='cpu')
    hp = ckpt['hyper_parameters']
    model = load_obj(hp['model_class'])(input_shape=tuple(hp['input_shape']), **hp['model_cls_args'])
    model.load_state_dict({k[len('model.'):]: v for k, v in ckpt['state_dict'].items() if k.startswith('model.')})
    model.tile_size = int(hp['input_shape'][1])
    return model.eval()


def make_normaliser(path):
    channels = yaml.safe_load(open(CHANNELS))['channels']
    spec = yaml.safe_load(open(path))
    return load_obj(spec['normaliser_cls'])(channels, **spec['normaliser_args'])


def to_tiles(x, t):
    c, h, w = x.shape
    rows, cols = h // t, w // t
    x = x[:, :rows * t, :cols * t].reshape(c, rows, t, cols, t)
    return x.transpose(1, 3, 0, 2, 4).reshape(rows * cols, c, t, t), (rows, cols)


@torch.no_grad()
def change_scores(model, before, after, batch=1024):
    out = []
    for i in range(0, len(before), batch):
        mu_before, _ = model.encode(torch.from_numpy(before[i:i + batch]))
        mu_after, _ = model.encode(torch.from_numpy(after[i:i + batch]))
        out.append((1 - cosine_similarity(mu_before, mu_after)).numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def prepare(path, bands, normaliser, calib):
    with rasterio.open(path) as src:
        x, profile = src.read(bands).astype(np.float32), src.profile
    if calib is not None:
        x = alignment.apply(x, calib['gains'], calib['offsets']).astype(np.float32)
    return normaliser(torch.from_numpy(x)).numpy(), profile


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('checkpoint')
    ap.add_argument('before')
    ap.add_argument('after')
    ap.add_argument('out')
    ap.add_argument('--normalisation', required=True)
    ap.add_argument('--bands', type=int, nargs=4, default=[3, 2, 1, 4])
    ap.add_argument('--alignment')
    a = ap.parse_args()
    model = load_model(a.checkpoint)
    t = model.tile_size
    normaliser = make_normaliser(a.normalisation)
    calib = yaml.safe_load(open(a.alignment)) if a.alignment else None
    before, profile = prepare(a.before, a.bands, normaliser, calib)
    after, profile_after = prepare(a.after, a.bands, normaliser, calib)
    if before.shape != after.shape or profile['transform'] != profile_after['transform']:
        raise SystemExit('the two scenes are not on the same grid')
    tiles_before, grid = to_tiles(before, t)
    tiles_after, _ = to_tiles(after, t)
    scores = change_scores(model, tiles_before, tiles_after).reshape(grid)
    profile.update(count=1, dtype='float32', height=grid[0], width=grid[1], nodata=None,
                   transform=profile['transform'] * rasterio.Affine.scale(t))
    with rasterio.open(a.out, 'w', **profile) as dst:
        dst.write(scores.astype(np.float32), 1)
    print(f'{grid[0]} x {grid[1]} tiles of {t} px -> {a.out}; '
          f'cosine distance median {np.median(scores):.4f}, max {scores.max():.4f}')


if __name__ == '__main__':
    main()
