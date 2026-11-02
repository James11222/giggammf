"""Test cubes shared by the test modules and the multi-process worker."""

from __future__ import annotations

import numpy as np


def lognormal_cube(n, seed=0):
    """A δ+1 field with cosmic-web-like contrast: exp of a red-spectrum Gaussian field."""
    rng = np.random.default_rng(seed)
    k = np.sqrt(sum(np.meshgrid(*[np.fft.fftfreq(n) ** 2] * 3, indexing="ij")))
    k[0, 0, 0] = 1.0
    amp = k**-1.5
    amp[0, 0, 0] = 0.0
    g = np.fft.ifftn(amp * np.fft.fftn(rng.standard_normal((n, n, n)))).real
    rho = np.exp(2.0 * g / g.std())
    return (rho / rho.mean()).astype(np.float32)
