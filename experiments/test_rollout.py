import numpy as np
import matplotlib.pyplot as plt
from envs.cartpole import CartPole

# --- setup ---
env = CartPole()  # use default parameters
rng = np.random.default_rng(0)
x = env.reset(rng)  # initial state
dt = 0.02  # 20 ms integration step
T = 5.0  # total simulation time
steps = int(T / dt)

# --- naive controller (for now, random force each step) ---
X = []
U = []
for _ in range(steps):
    if env.done(x):
        break
    u = np.array([rng.uniform(-env.Fmax, env.Fmax)])  # random action
    X.append(x.copy())
    U.append(u.copy())
    x = env.step(x, u, dt)

X = np.array(X)
U = np.array(U)

# --- visualize ---
t = np.arange(len(X)) * dt
fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
ax[0].plot(t, X[:, 0], label="Cart position (m)")
ax[0].plot(t, X[:, 2], label="Pole angle (rad)")
ax[0].legend()
ax[0].set_ylabel("State")

ax[1].plot(t, U[:, 0], label="Force (N)")
ax[1].legend()
ax[1].set_ylabel("Action")
ax[1].set_xlabel("Time (s)")
plt.tight_layout()
plt.show()
