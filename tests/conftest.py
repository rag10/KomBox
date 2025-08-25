from __future__ import annotations
import sys, pathlib

# Garantiza que 'kombox' se pueda importar tanto si ejecutas pytest desde la raíz
# como desde dentro del subdirectorio 'kombox/'.
try:
    import kombox  # noqa: F401
except Exception:
    here = pathlib.Path(__file__).resolve()
    for p in [here.parents[i] for i in range(1, 6)]:
        if (p / "kombox" / "__init__.py").exists():
            sys.path.insert(0, str(p))
            break
