"""python tests/test_revision.py"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data import alignment  # noqa: E402
from src.data.normalisers import percentile_spec  # noqa: E402
from src.data.transformations import RandomResample  # noqa: E402
from src.models.ae_vae_models.deeper_vae import ResolutionAdaptiveVAE  # noqa: E402
from src.utils import load_obj  # noqa: E402

torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', min(4, os.cpu_count() or 1))))
fails = 0
ARGS = dict(hidden_channels=[32, 64, 128, 256], latent_dim=128, extra_depth_on_scale=2, visualisation_channels=[0, 1, 2])
CLASS = 'src.models.ae_vae_models.deeper_vae.ResolutionAdaptiveVAE'


def check(ok, what):
    global fails
    print(('ok   ' if ok else 'FAIL ') + what)
    fails += not ok


def save_checkpoint(model, path, args=ARGS):
    torch.save({'hyper_parameters': {'model_class': CLASS, 'input_shape': (4, 32, 32), 'model_cls_args': args},
                'state_dict': {'model.' + k: v for k, v in model.state_dict().items()}}, path)


torch.manual_seed(0)
model = ResolutionAdaptiveVAE((4, 32, 32), **ARGS)
x = torch.randn(8, 4, 32, 32)
recon, mu, log_var = model(x)
check(recon.shape == (8, 4, 32, 32) and mu.shape == (8, 128) and log_var.shape == (8, 128),
      'the model runs on 4-band 32 x 32 tiles: reconstruction, and a 128-dimensional latent mean and log-variance')
check(isinstance(model.pool, torch.nn.AdaptiveAvgPool2d) and model.fc_mu.in_features == 256,
      'the final feature map is globally pooled before the latent projection')

loss = model.loss_function(x, [recon, mu, log_var], M_N=1e-4)
kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1))
check(torch.allclose(loss['loss'], ((recon - x) ** 2).sum() / 8 + kld),
      'the loss is the per-tile summed squared error + KL, beta = 1, whatever M_N the module passes')


def smooth(n, seed):
    g = torch.Generator().manual_seed(seed)
    z = torch.nn.functional.avg_pool2d(torch.randn(n, 4, 40, 40, generator=g), 9, stride=1)[..., :32, :32]
    return torch.tanh(z * (1 + 3 * torch.rand(n, 1, 1, 1, generator=g)) + 0.5 * torch.randn(n, 4, 1, 1, generator=g))


torch.manual_seed(2)
trained = ResolutionAdaptiveVAE((4, 32, 32), **ARGS)
opt = torch.optim.Adam(trained.parameters(), lr=1e-3)
data = smooth(256, 3)
losses = []
for _ in range(120):
    batch = data[torch.randint(0, 256, (32,))]
    out_ = trained(batch)
    loss_ = trained.loss_function(batch, out_)
    opt.zero_grad()
    loss_['loss'].backward()
    opt.step()
    losses.append(loss_['loss'].item())
trained.eval()
with torch.no_grad():
    held = smooth(64, 4)
    mu_held = trained.encode(held)[0]
    kl = torch.mean(-0.5 * torch.sum(1 + trained.encode(held)[1] - mu_held ** 2 - trained.encode(held)[1].exp(), dim=1)).item()
    mse = torch.nn.functional.mse_loss(trained.decode(mu_held), held).item()
check(losses[-1] < 0.5 * losses[0] and mse < 0.8 * held.var().item() and kl > 1,
      f'training reduces the loss ({losses[0]:.0f} -> {losses[-1]:.0f}) and the latent stays informative '
      f'(KL {kl:.1f} nats; reconstruction error {mse:.3f} below the tiles\' variance {held.var().item():.3f})')

sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import infer  # noqa: E402

with tempfile.TemporaryDirectory() as d:
    ckpt = os.path.join(d, 'model.ckpt')
    save_checkpoint(model, ckpt)
    loaded = infer.load_model(ckpt)
    same = all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), loaded.state_dict().values()))
    m1, m2 = loaded.encode(x)[0], loaded.encode(x)[0]
    fresh = ResolutionAdaptiveVAE((4, 32, 32), **ARGS).eval()
    check(same and torch.equal(m1, m2) and not torch.allclose(fresh.encode(x)[0], m1),
          'a checkpoint loads exactly, repeated runs give identical output, and a fresh random model differs')
    bad = os.path.join(d, 'bad.ckpt')
    save_checkpoint(model, bad, {**ARGS, 'latent_dim': 100})
    try:
        infer.load_model(bad)
        check(False, 'a checkpoint that does not match the architecture is an error')
    except RuntimeError:
        check(True, 'a checkpoint that does not match the architecture is an error')

a_, b_ = held[:32].numpy(), held[32:].numpy()
same_score = infer.change_scores(trained, a_, a_)
diff_score = infer.change_scores(trained, a_, b_)
check(np.allclose(same_score, 0, atol=1e-5) and np.median(diff_score) > 0.05,
      f'identical tiles score zero; different tiles do not (median {np.median(diff_score):.3f})')

rng = np.random.default_rng(1)
bands = ['B4', 'B3', 'B2', 'B8']
raw = [rng.lognormal(7, 0.4, 50_000) for _ in bands]
spec = percentile_spec([[np.log(v + 1.0)] for v in raw], bands)
norm = load_obj(spec['normaliser_cls'])(bands, **spec['normaliser_args'])
p1 = [np.exp(np.percentile(np.log(v + 1.0), 1)) - 1.0 for v in raw]
p99 = [np.exp(np.percentile(np.log(v + 1.0), 99)) - 1.0 for v in raw]
img = torch.tensor(np.stack([np.array([[a, b, 0.0]]) for a, b in zip(p1, p99)]), dtype=torch.float32)
out = norm(img.clone())
check(np.allclose(out[:, 0, 0], -1, atol=1e-3) and np.allclose(out[:, 0, 1], 1, atol=1e-3) and torch.isnan(out[:, 0, 2]).all(),
      'normalisation: log(x + eps), then P1 -> -1 and P99 -> +1 per band; values <= 0 stay nodata')

truth = rng.normal(0.2, 0.05, 20_000)
planet = truth + rng.normal(0, 0.02, truth.size)
sentinel = 1.3 * truth + 0.05 + rng.normal(0, 0.02, truth.size)
gain, offset = alignment.major_axis_fit(planet, sentinel)
ols = np.polyfit(planet, sentinel, 1)[0]
check(abs(gain - 1.3) < 0.03 and abs(offset - 0.05) < 0.01 and abs(ols - 1.3) > abs(gain - 1.3),
      f'major-axis regression recovers gain 1.3 and offset 0.05 ({gain:.3f}, {offset:.3f}); least squares is biased ({ols:.3f})')

tile = torch.randn(2, 4, 32, 32)
tile[0, :, :4, :4] = float('nan')
zoomed = RandomResample(3.33, 1.0)(tile)
check(torch.equal(RandomResample(1.0, 1.0)(tile).nan_to_num(), tile.nan_to_num())
      and torch.equal(RandomResample(3.33, 0.0)(tile).nan_to_num(), tile.nan_to_num())
      and zoomed.shape == tile.shape and not torch.allclose(zoomed.nan_to_num(), tile.nan_to_num()),
      'scale augmentation: identity at zoom 1 or p = 0; otherwise a resampled tile of the same size')

try:
    import rasterio
    from rasterio.transform import from_origin
except ImportError:
    rasterio = None
if rasterio is None:
    print('skip end-to-end inference: rasterio not installed')
else:
    with tempfile.TemporaryDirectory() as d:
        save_checkpoint(trained, os.path.join(d, 'm.ckpt'))
        scene = smooth(9, 5).numpy().reshape(3, 3, 4, 32, 32).transpose(2, 0, 3, 1, 4).reshape(4, 96, 96)
        before = np.exp(7 + scene).astype('uint16')
        after = before.copy()
        after[:, 32:64, 32:64] = (before[:, 32:64, 32:64] * 0.4).astype('uint16')
        prof = dict(driver='GTiff', height=96, width=96, count=4, dtype='uint16', crs='EPSG:32635',
                    transform=from_origin(500000, 1600000, 3, 3))
        for name, arr in (('before', before), ('after', after)):
            with rasterio.open(os.path.join(d, f'{name}.tif'), 'w', **prof) as dst:
                dst.write(arr)
        norm_file = os.path.join(d, 'norm.yaml')
        yaml.safe_dump(percentile_spec([[np.log(before[b].ravel() + 1.0)] for b in range(4)], bands), open(norm_file, 'w'))
        p = subprocess.run([sys.executable, '-m', 'scripts.infer', os.path.join(d, 'm.ckpt'), os.path.join(d, 'before.tif'),
                            os.path.join(d, 'after.tif'), os.path.join(d, 'out.tif'), '--normalisation', norm_file,
                            '--bands', '1', '2', '3', '4'], cwd=ROOT, capture_output=True, text=True)
        ok = p.returncode == 0
        if ok:
            with rasterio.open(os.path.join(d, 'out.tif')) as src:
                cm, tr = src.read(1), src.transform
            ok = cm.shape == (3, 3) and np.allclose(np.delete(cm.ravel(), 4), 0, atol=1e-5) and cm[1, 1] > 0.01 and tr.a == 96
        check(ok, 'end to end: a 3 x 3 tile map, zero where the scenes agree, the changed tile flagged, 96 m per tile'
              + ('' if p.returncode == 0 else f'\n{p.stderr[-800:]}'))

if rasterio is not None:
    with tempfile.TemporaryDirectory() as d:
        save_checkpoint(trained, os.path.join(d, 'm.ckpt'))
        scene = smooth(36, 6).numpy().reshape(6, 6, 4, 32, 32).transpose(2, 0, 3, 1, 4).reshape(4, 192, 192)
        pre = np.exp(7 + scene)
        post = pre * rng.uniform(0.98, 1.02, pre.shape)
        labels = np.zeros((1, 192, 192), 'uint8')
        for r, c in ((0, 1), (1, 4), (2, 2), (3, 5), (4, 0), (5, 3)):
            post[:, 32 * r:32 * r + 32, 32 * c:32 * c + 32] *= 0.4
            labels[0, 32 * r:32 * r + 32, 32 * c:32 * c + 32] = 1
        prof = dict(driver='GTiff', height=192, width=192, crs='EPSG:32635', transform=from_origin(500000, 1600000, 3, 3))
        for name, arr in (('pre', pre), ('pre2', pre * 1.01), ('post', post), ('labels', labels)):
            arr = arr if name == 'labels' else arr.astype('uint16')
            with rasterio.open(os.path.join(d, f'{name}.tif'), 'w', count=len(arr), dtype=arr.dtype, **prof) as dst:
                dst.write(arr)
        norm_file = os.path.join(d, 'norm.yaml')
        yaml.safe_dump(percentile_spec([[np.log(pre[b].ravel() + 1.0)] for b in range(4)], bands), open(norm_file, 'w'))
        ok = True
        for pres in (['pre.tif'], ['pre.tif', 'pre2.tif']):
            out = os.path.join(d, f'out{len(pres)}')
            p = subprocess.run([sys.executable, '-m', 'scripts.evaluate', os.path.join(d, 'post.tif'),
                                os.path.join(d, 'labels.tif'), out, '--pre', *[os.path.join(d, f) for f in pres],
                                '--checkpoint', os.path.join(d, 'm.ckpt'),
                                '--normalisation', norm_file, '--bands', '1', '2', '3', '4', '--red', '1', '--nir', '4',
                                '--n-boot', '50'], cwd=ROOT, capture_output=True, text=True)
            if p.returncode != 0:
                ok = False
                print(p.stderr[-800:])
                break
            import pandas as pd
            table = pd.read_csv(os.path.join(out, 'metrics.csv')).set_index('method')
            ok &= (len(table) == 9 and table.loc['LRC', 'AUPRC'] > 0.9 and table[['AUPRC', 'F1']].notna().all().all()
                   and len(pd.read_csv(os.path.join(out, 'effects.csv'))) == 8
                   and os.path.exists(os.path.join(out, 'spearman.csv')))
        check(ok, 'evaluation, single scene and time series: nine methods scored, burned tiles ranked first by LRC, '
              'effect sizes and Spearman correlation written')

try:
    import pytorch_lightning  # noqa: F401
    from src.models.module import Module
    cfg = {'model_class': CLASS, 'model_cls_args': dict(ARGS), 'input_shape': (4, 32, 32),
           'len_train_ds': 1000, 'len_val_ds': 100}
    train_cfg = {'lr': 1e-3, 'weight_decay': 0.0, 'da_trans_cls': 'src.data.transformations.Transformer',
                 'da_trans_args': [{'src.data.transformations.RandomResample': {'transf_args': {'max_zoom': 3.33, 'p': 1.0}}}]}
    module = Module(cfg, train_cfg)
    tl = module.training_step([torch.randn(4, 4, 32, 32)], 0)
    vl = module.validation_step([torch.randn(4, 4, 32, 32)], 0)
    check(type(module.model).__name__ == 'ResolutionAdaptiveVAE' and torch.isfinite(tl['loss']) and torch.isfinite(vl['loss']),
          "the training module builds the paper's model from the config, and trains and validates")
except ImportError:
    print('skip module check: pytorch_lightning not installed')

try:
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=os.path.join(ROOT, 'config')):
        cfg = compose('config.yaml', overrides=['+dataset=alpha_multiscene', '+channels=rgb_nir', '+normalisation=raw',
                                                '+module=deeper_vae', '+training=da', '+transform=scale', '+project=x'])
    m, tr = cfg.module, cfg.training
    check(m.model_class.endswith('ResolutionAdaptiveVAE') and m.model_cls_args.latent_dim == 128 and tr.epochs == 200
          and tr.batch_size_train == 128 and 'RandomResample' in str(tr.da_trans_args),
          'the training config composes: the paper model, latent 128, 200 epochs, batch 128, scale augmentation')
except ImportError:
    print('skip config check: hydra not installed')

print('all checks passed' if not fails else f'{fails} checks failed')
sys.exit(1 if fails else 0)
