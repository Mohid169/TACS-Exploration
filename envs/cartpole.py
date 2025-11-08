import numpy as np
from .base import Env, Array


class CartPole(Env):
    state_size = 4
    action_size = 1

    def __init__(
        self,
        mc: float = 1.0,  # cart mass
        mp: float = 0.1,  # pole mass
        l: float = 0.5,  # pole length (hinge to COM)
        Fmax: float = 30.0,  # actuator limit
        L: float = 2.0,
    ):  # track length (termination only)
        self.mc, self.mp, self.l = mc, mp, l
        self.Fmax, self.L = Fmax, L

    def reset(self, rng: np.random.Generator) -> Array:
        th0 = rng.uniform(-0.2, 0.2)
        return np.array([0.0, 0.0, th0, 0.0], Dtype=float)  # [x, xdot, th, thdot]

    def step(self, x: Array, u: Array, dt: float) -> Array:
        mc, mp, l = self.mc, self.mp, self.l
        x_, xd, th, thd = x
        # clip action
        u = float(np.clip(u, -self.Fmax, self.Fmax))

        # dynamics (standard cart-pole, continuous force)
        temp = (u + mp * l * thd**2 * np.sin(th)) / (mc + mp)
        thdd = (g * np.sin(th) - np.cos(th) * temp) / (
            l * (4.0 / 3.0 - (mp * (np.cos(th) ** 2)) / (mc + mp))
        )
        xdd = temp - (mp * l * thdd * np.cos(th)) / (mc + mp)

        # semi-implicit Euler
        xd_new = xd + dt * xdd
        x_new = x_ + dt * xd_new
        thd_new = thd + dt * thdd
        th_new = th + dt * thd_new

        return np.array([x_new, xd_new, th_new, thd_new], dtype=float)

    def done(self, x: Array) -> bool:
        #Stop the simulation if the pole falls beyond 60° or if the cart slides off the track.”
        th, xc = x[2], x[0]
        return (abs(th) > np.deg2rad(60.0)) or (abs(xc) > self.L / 2)
