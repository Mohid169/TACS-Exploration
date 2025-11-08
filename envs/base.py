from __future__ import annotations
import numpy as np

Array = np.ndarray

class Env:
    "interface for one-step simulations"
    state_size: int
    action_size: int = 1

    def reset(self, rng: np.random.Generator) -> Array:
        raise NotImplementedError

    def step(self, x: Array, u: Array, dt: float) -> Array:
        """Return next state given current state x and action u."""
        raise NotImplementedError

    def done(self, x: Array) -> bool:
        """Episode termination condition."""
        raise NotImplementedError
