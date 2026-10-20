
import torch
import torch.nn.functional as F

from src.utils import load_obj

class RandomResample(torch.nn.Module):
    """With probability p, zoom each tile by a random factor in [1, max_zoom]."""
    def __init__(self, max_zoom: float, p: float):
        super().__init__()
        assert max_zoom >= 1, "max_zoom should be at least 1"
        assert 0 <= p <= 1, "p should be between 0 and 1"
        self.max_zoom = float(max_zoom)
        self.p = float(p)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single = x.dim() == 3
        x = x.unsqueeze(0) if single else x
        size = x.shape[-1]
        out = x.clone()
        for i in range(x.shape[0]):
            if torch.rand(()) >= self.p:
                continue
            zoom = 1 + torch.rand(()).item() * (self.max_zoom - 1)
            crop = max(2, int(round(size / zoom)))
            top, left = (torch.randint(0, size - crop + 1, (2,))).tolist()
            patch = x[i:i + 1, :, top:top + crop, left:left + crop]
            nodata = torch.isnan(patch).float()
            resampled = F.interpolate(torch.nan_to_num(patch), size=(size, size), mode='bilinear', align_corners=False)
            resampled[F.interpolate(nodata, size=(size, size), mode='nearest') > 0] = float('nan')
            out[i] = resampled[0]
        return out[0] if single else out


class Transformer(torch.nn.Module):

    def __init__(self, *args):
        super().__init__()

        transforms = []
        for t in args:
            for t_cls, t_args in t.items():
                transforms.append(load_obj(t_cls)(**t_args["transf_args"]))

        self.transforms = torch.nn.Sequential(*transforms)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.transforms(x)
        return x_out

class NoTransformer(torch.nn.Module):

    def __init__(self, *args):
        super().__init__()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
