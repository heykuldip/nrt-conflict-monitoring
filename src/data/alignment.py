"""Per-band gain and offset by major-axis regression (Eq. 1)."""
import numpy as np


def major_axis_fit(x, y):
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    sxx, syy = x.var(), y.var()
    sxy = ((x - x.mean()) * (y - y.mean())).mean()
    gain = (syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) / (2 * sxy)
    return float(gain), float(y.mean() - gain * x.mean())


def fit_bands(planet, sentinel):
    return [major_axis_fit(p, s) for p, s in zip(planet, sentinel)]


def apply(image, gains, offsets):
    return image * np.asarray(gains, float)[:, None, None] + np.asarray(offsets, float)[:, None, None]
