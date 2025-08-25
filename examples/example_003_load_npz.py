# examples/example_003_load_npz.py
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from kombox.core.utils import load_npz_series
import matplotlib.pyplot as plt

def main():
    # Si grabaste con: path_pattern="examples/msd_trace_part{part:04d}.npz"
    data = load_npz_series(os.path.join(os.path.dirname(__file__), "msd_trace_part_{part:04d}.npz"))
    print("Claves:", list(data.keys()))
    print("t shape:", data["t"].shape)
    print("mass.x shape:", data["mass.x"].shape)      # (T_total, B, 1)
    print("sum.F shape:", data["sum.F"].shape)        # (T_total, B, 1)

    # También puedes obtener tensores de torch directamente:
    # data_t = load_npz_series("...{part:04d}.npz", to_torch=True)
    # print(data_t["mass.x"].device, data_t["mass.x"].dtype)

    
    t = data["t"]
    x = data["mass.x"]
    plt.plot(t, x[:,0], label="x (batch 0)")
    plt.plot(t, x[:,1], label="x (batch 1)")
    plt.plot(t, x[:,2], label="x (batch 1)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
