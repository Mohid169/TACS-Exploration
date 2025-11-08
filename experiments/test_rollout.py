# experiments/test_rollout.py
# Run from repo root:  python -m experiments.test_rollout
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from envs.cartpole import CartPole
from tacs.visualize import plot_timeseries, animate_cartpole

def rollout(env, K, dt: float, T: float, rng=None):
    rng = rng or np.random.default_rng(0)
    x = env.reset(rng)
    X, U = [], []
    steps = int(T / dt)
    for k in range(steps):
        if env.done(x):
            break
        u = np.array([K @ x])
        u = np.clip(u, -env.Fmax, env.Fmax)
        X.append(x.copy()); U.append(u.copy())
        x = env.step(x, u, dt)
    t = np.arange(len(X)) * dt
    return np.asarray(X), np.asarray(U), t

def main():
    env = CartPole()                 # defaults are fine
    K = np.array([1.0, 1.5, 25.0, 3.5])   # stabilizing gains you used

    dt, T = 0.02, 5.0
    X, U, t = rollout(env, K, dt, T)

    plot_timeseries(X, U, t)
    animate_cartpole(L=env.L, pole_len=env.l, X=X, t=t)

if __name__ == "__main__":
    main()
