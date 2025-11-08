import numpy as np
import matplotlib.pyplot as plt
from envs.cartpole import CartPole
import matplotlib.animation as animation


# --- setup ---
env = CartPole()  # use default parameters
rng = np.random.default_rng(0)
x = env.reset(rng)  # initial state
dt = 0.02  # 20 ms integration step
T = 5.0  # total simulation time
steps = int(T / dt)


# implemneting a small linear controller
K = np.array([1.0, 1.5, 25.0, 3.5])  # tuned gains
X = []
U = []
for _ in range(steps):
    if env.done(x):
        break
    u = np.array([K @ x])  # random action
    u = np.clip(u, -env.Fmax, env.Fmax)
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

def animate_cartpole(X, dt):
    # unpack
    x = X[:, 0]    # cart position
    th = X[:, 2]   # pole angle

    # geometry constants
    l = env.l        # pole length
    cart_w = 0.2
    cart_h = 0.1

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(-env.L/2 - 0.5, env.L/2 + 0.5)
    ax.set_ylim(-l - 0.1, l + 0.1)
    ax.set_xlabel("Track position (m)")
    ax.set_ylabel("Height (m)")
    ax.set_aspect('equal', 'box')

    # static base line
    ax.plot([-env.L/2, env.L/2], [0, 0], "k-", lw=2)

    # elements to animate
    cart_patch = plt.Rectangle((0, 0), cart_w, cart_h, color="tab:blue", ec="k")
    pole_line, = ax.plot([], [], "o-", color="tab:orange", lw=2, markersize=6)
    ax.add_patch(cart_patch)

    def init():
        cart_patch.set_xy((-cart_w/2, -cart_h/2))
        pole_line.set_data([], [])
        return cart_patch, pole_line

    def update(frame):
        # cart coordinates
        xc = x[frame]
        yc = 0
        cart_patch.set_xy((xc - cart_w/2, yc - cart_h/2))

        # pole end
        xp = xc + l * np.sin(th[frame])
        yp = yc + l * np.cos(th[frame])
        pole_line.set_data([xc, xp], [yc, yp])
        return cart_patch, pole_line

    ani = animation.FuncAnimation(fig, update, frames=len(x), init_func=init,
                                  blit=True, interval=dt*1000)
    plt.show()

animate_cartpole(X, dt)