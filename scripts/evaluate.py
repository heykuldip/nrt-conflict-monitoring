"""
python -m scripts.evaluate POST.tif LABELS.tif OUT_DIR --pre PRE.tif [PRE.tif ...] [--checkpoint CKPT --normalisation FILE.yaml]
    [--alignment FILE.yaml] [--bands 3 2 1 4] [--red 3 --nir 4] [--scale 10000] [--n-boot 1000]
Several --pre scenes: the reference is their mean (pixels and indices) or the mean of their latent means.
LABELS.tif: fire-affected pixels > 0; a tile is fire-affected if any of its pixels is.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import yaml
from scipy.stats import chi2, spearmanr, wilcoxon
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from scripts.infer import load_model, make_normaliser, to_tiles
from src.data import alignment

METRICS = ['AUPRC', 'Precision', 'Recall', 'F1', 'CE', 'OE']


def read(path, bands):
    with rasterio.open(path) as src:
        return src.read(bands).astype(np.float32), src.transform


def pixel_cosine(pre, post):
    a, b = pre.reshape(len(pre), -1), post.reshape(len(post), -1)
    return 1 - (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def cva(pre, post):
    return np.sqrt(((post - pre) ** 2).sum(1)).mean((1, 2))


def irmad(pre, post, iters=50, tol=1e-6):
    n, c, t, _ = pre.shape
    x = pre.transpose(0, 2, 3, 1).reshape(-1, c).astype(float)
    y = post.transpose(0, 2, 3, 1).reshape(-1, c).astype(float)
    w, rho_old = np.ones(len(x)), np.zeros(c)
    for _ in range(iters):
        ws = (w / w.sum())[:, None]
        xc, yc = x - (ws * x).sum(0), y - (ws * y).sum(0)
        lx = np.linalg.inv(np.linalg.cholesky((ws * xc).T @ xc))
        ly = np.linalg.inv(np.linalg.cholesky((ws * yc).T @ yc))
        u, rho, vt = np.linalg.svd(lx @ ((ws * xc).T @ yc) @ ly.T)
        mad = xc @ (lx.T @ u) - yc @ (ly.T @ vt.T)
        z = (mad ** 2 / np.maximum(2 * (1 - rho), 1e-12)).sum(1)
        w = 1 - chi2.cdf(z, c)
        if np.abs(rho - rho_old).max() < tol:
            break
        rho_old = rho
    return z.reshape(n, t * t).mean(1)


def ndvi(red, nir):
    return (nir - red) / (nir + red + 1e-9)


def bai(red, nir):
    return 1 / np.maximum((0.1 - red) ** 2 + (0.06 - nir) ** 2, 1e-6)  # finite at the charcoal point


@torch.no_grad()
def latents(model, tiles, batch=1024):
    out = [model.encode(torch.from_numpy(tiles[i:i + batch])) for i in range(0, len(tiles), batch)]
    return np.concatenate([m.numpy() for m, _ in out]), np.exp(np.concatenate([v.numpy() for _, v in out]))


def latent_scores(mu_pre, var_pre, mu_post, var_post):
    diff = mu_post - mu_pre
    cos = 1 - (mu_pre * mu_post).sum(1) / (np.linalg.norm(mu_pre, axis=1) * np.linalg.norm(mu_post, axis=1) + 1e-12)
    cov = np.cov(np.vstack([mu_pre, mu_post]).T) + 1e-6 * np.eye(mu_pre.shape[1])
    maha = np.sqrt(np.einsum('ij,jk,ik->i', diff, np.linalg.inv(cov), diff))
    mid_mu, mid_var = (mu_pre + mu_post) / 2, (var_pre + var_post) / 2 + (diff / 2) ** 2

    def kl(m, v):
        return 0.5 * (np.log(mid_var / v) + (v + (m - mid_mu) ** 2) / mid_var - 1).sum(1)

    return {'LRC': cos, 'LRC Euclidean': np.linalg.norm(diff, axis=1), 'LRC Mahalanobis': maha,
            'LRC JS divergence': 0.5 * (kl(mu_pre, var_pre) + kl(mu_post, var_post))}


def metrics(y, s, q=95):
    if y.min() == y.max():
        return [np.nan] * len(METRICS)
    p, r, f, _ = precision_recall_fscore_support(y, s > np.percentile(s, q), average='binary', zero_division=0)
    return [average_precision_score(y, s), p, r, f, 1 - p, 1 - r]


def bootstrap(y, scores, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    return {m: np.array([metrics(y[i], s[i]) for i in idx]) for m, s in scores.items()}


def summarise(boot):
    rows = []
    for m, b in boot.items():
        row = {'method': m}
        for k, med, lo, hi in zip(METRICS, np.nanmedian(b, 0), np.nanpercentile(b, 2.5, 0), np.nanpercentile(b, 97.5, 0)):
            row.update({k: med, f'{k} lo': lo, f'{k} hi': hi})
        rows.append(row)
    return pd.DataFrame(rows)


def effects(boot, reference='LRC'):
    rows = []
    for m, b in boot.items():
        if m == reference:
            continue
        d = boot[reference] - b
        auprc = d[~np.isnan(d[:, 0]), 0]
        p = wilcoxon(auprc).pvalue if np.any(auprc != 0) else 1.0
        cohen = np.nanmean(d[:, :4], 0) / np.nanstd(d[:, :4], 0, ddof=1)
        rows.append({'method': m, 'Wilcoxon p (AUPRC)': p, **{f"Cohen's d ({k})": v for k, v in zip(METRICS[:4], cohen)}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('post')
    ap.add_argument('labels')
    ap.add_argument('out')
    ap.add_argument('--pre', nargs='+', required=True)
    ap.add_argument('--checkpoint')
    ap.add_argument('--normalisation')
    ap.add_argument('--alignment')
    ap.add_argument('--bands', type=int, nargs='+', default=[3, 2, 1, 4])
    ap.add_argument('--red', type=int, default=3)
    ap.add_argument('--nir', type=int, default=4)
    ap.add_argument('--scale', type=float, default=10000)
    ap.add_argument('--tile', type=int, default=32)
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if a.checkpoint and not a.normalisation:
        ap.error('--checkpoint needs --normalisation')

    model = load_model(a.checkpoint) if a.checkpoint else None
    t = model.tile_size if model else a.tile
    normaliser = make_normaliser(a.normalisation) if a.normalisation else None
    calib = yaml.safe_load(open(a.alignment)) if a.alignment else None

    def prepare(path):
        x, transform = read(path, a.bands)
        if calib is not None:
            x = alignment.apply(x, calib['gains'], calib['offsets']).astype(np.float32)
        x = np.nan_to_num(normaliser(torch.from_numpy(x)).numpy()) if normaliser else x / a.scale
        return to_tiles(x, t)[0], transform

    post, grid = prepare(a.post)
    pres = [prepare(p) for p in a.pre]
    labels, label_grid = read(a.labels, [1])
    tiles, (rows, cols) = to_tiles(labels, t)
    if any(p.shape != post.shape or g != grid for p, g in pres) or label_grid != grid or len(tiles) != len(post):
        raise SystemExit('the scenes and labels are not on the same grid')
    pre = np.mean([p for p, _ in pres], 0)
    y = (tiles > 0).any((1, 2, 3)).astype(int)

    def index_tiles(path, fn):
        red_nir = read(path, [a.red, a.nir])[0] / a.scale
        return np.nanmean(to_tiles(fn(*red_nir)[None], t)[0], (1, 2, 3))

    scores = {'Cosine distance': pixel_cosine(pre, post), 'CVA': cva(pre, post), 'IR-MAD': irmad(pre, post),
              'dNDVI': np.mean([index_tiles(p, ndvi) for p in a.pre], 0) - index_tiles(a.post, ndvi),
              'dBAI': index_tiles(a.post, bai) - np.mean([index_tiles(p, bai) for p in a.pre], 0)}
    if model is not None:
        pre_latents = [latents(model, p) for p, _ in pres]
        mu_post, var_post = latents(model, post)
        scores.update(latent_scores(np.mean([m for m, _ in pre_latents], 0), np.mean([v for _, v in pre_latents], 0),
                                    mu_post, var_post))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'row': np.repeat(np.arange(rows), cols), 'col': np.tile(np.arange(cols), rows), 'label': y, **scores}).to_csv(
        out / 'scores.csv', index=False)
    valid = np.all([np.isfinite(s) for s in scores.values()], 0)
    y, scores = y[valid], {m: s[valid] for m, s in scores.items()}
    boot = bootstrap(y, scores, a.n_boot, a.seed)
    table = summarise(boot)
    table.to_csv(out / 'metrics.csv', index=False, float_format='%.4f')
    if model is not None:
        effects(boot).to_csv(out / 'effects.csv', index=False, float_format='%.4g')
        rho, p = spearmanr(scores['LRC'], scores['dBAI'])
        pd.DataFrame([{'Spearman rho (LRC, dBAI)': rho, 'p': p}]).to_csv(out / 'spearman.csv', index=False, float_format='%.4g')
    print(f'{len(y)} tiles ({(~valid).sum()} without a score excluded), {y.sum()} fire-affected; '
          f'{a.n_boot} bootstrap resamples -> {out}')
    print(table[['method'] + METRICS].to_string(index=False, float_format='%.2f'))


if __name__ == '__main__':
    main()
