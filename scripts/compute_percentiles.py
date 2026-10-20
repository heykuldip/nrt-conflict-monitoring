"""
python -m scripts.compute_percentiles +dataset=alpha_multiscene +channels=rgb_nir +normalisation=raw \
        +training=da +transform=no_augmentation +out=config/normalisation/log_percentiles.yaml
"""
import hydra
import numpy as np
import yaml
from hydra.utils import to_absolute_path

from src.utils import deepconvert
from src.data.datamodule import ParsedDataModule
from src.data.normalisers import percentile_spec

EPS = 1.0
PER_BATCH = 20_000


@hydra.main(config_path='../config', config_name='config.yaml')
def main(cfg):
    cfg = deepconvert(cfg)
    channels = cfg['channels']['channels']
    data_module = ParsedDataModule.load_or_create(cfg['dataset'], cfg['cache_dir'])
    loader = data_module.train_dataloader()
    loader = loader[0] if isinstance(loader, list) else loader
    rng = np.random.default_rng(0)
    samples = [[] for _ in channels]
    for k, batch in enumerate(loader):
        x = batch[0].numpy()
        for b in range(len(channels)):
            v = x[:, b].ravel()
            v = v[np.isfinite(v) & (v > 0)]
            if len(v) > PER_BATCH:
                v = rng.choice(v, PER_BATCH, replace=False)
            samples[b].append(np.log(v + EPS))
        if k + 1 >= cfg.get('max_batches', 500):
            break
    out = to_absolute_path(cfg['out'])
    with open(out, 'w') as f:
        f.write('---\n')
        yaml.safe_dump(percentile_spec(samples, channels), f, sort_keys=False)
    print(f'{k + 1} batches -> {out}')


if __name__ == '__main__':
    main()
