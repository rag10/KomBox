# kombox/utils/plotting.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def plot_series_single(ax, t: np.ndarray, y: np.ndarray, label: str, lw: float = 1.5):
    if y.ndim == 1:
        ax.plot(t, y, label=label, linewidth=lw)
    else:
        n = y.shape[1]
        for i in range(n):
            lab = f"{label}[{i}]" if n > 1 else label
            ax.plot(t, y[:, i], label=lab, linewidth=lw)
    ax.grid(True, alpha=0.3)

def plot_series_batch(ax, t: np.ndarray, y: np.ndarray, idx: int = 0, label: str = "", lw: float = 1.8):
    sel = y[:, idx, :]
    plot_series_single(ax, t, sel.squeeze(-1) if sel.shape[1] == 1 else sel, label=label, lw=lw)

def plot_series_mean_std(ax, t: np.ndarray, y: np.ndarray, label: str = "", lw: float = 2.0, alpha: float = 0.15):
    assert y.ndim == 3 and y.shape[2] == 1, "plot_series_mean_std asume n=1"
    mu = y.mean(axis=1)[:, 0]; sd = y.std(axis=1)[:, 0]
    ax.plot(t, mu, label=label, linewidth=lw)
    ax.fill_between(t, mu - sd, mu + sd, alpha=alpha)
    ax.grid(True, alpha=0.3)
