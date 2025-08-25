# kombox/__init__.py
from __future__ import annotations

__all__ = [
    "__version__",
    "Block","ContinuousBlock","DiscreteBlock","PortSpec",
    "Model","Simulator",
    "SolverBase","EulerSolver","RK4Solver","RK45Solver","TorchDiffEqSolver",
    "MemoryRecorder","NPZChunkRecorder",
    "list_npz_parts","load_npz_series",
    "Adder","Gain","Constant","Passthrough",
    "Mass1D","Spring","Damper",
]


__version__ = "0.1.0"

# Core
from .core.block import Block, ContinuousBlock, DiscreteBlock, PortSpec
from .core.model import Model
from .core.simulator import Simulator

# Solvers
from .core.solvers import SolverBase, EulerSolver, RK4Solver, RK45Solver, TorchDiffEqSolver

# Recorders y utils
from .core.recorders import MemoryRecorder, NPZChunkRecorder
from .core.utils import list_npz_parts, load_npz_series

# Bloques de ejemplo
from .blocks.basic import Adder, Gain, Constant
from .blocks.mechanical import Mass1D, Spring, Damper
