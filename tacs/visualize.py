# tacs/visualize.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_timeseries(X: np.ndarray, U: np.ndarray, t: np.ndarray) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    ax[0].plot(t, X[:, 0], label="Cart position (m)")
    ax[0].plot(t, X[:, 2], label="Pole angle (rad)")
    ax[0].legend(); ax[0].set_ylabel("State")

    ax[1].plot(t, U[:, 0], label="Force (N)")
    ax[1].legend(); ax[1].set_ylabel("Action"); ax[1].set_xlabel("Time (s)")
    plt.tight_layout(); plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_cartpole(L: float, pole_len: float, X: np.ndarray, t: np.ndarray):
    x = X[:, 0]
    th = X[:, 2]
    cart_w, cart_h = 0.20, 0.10

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(-L/2 - 0.5, L/2 + 0.5)
    ax.set_ylim(-pole_len - 0.1, pole_len + 0.1)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("Track (m)")
    ax.plot([-L/2, L/2], [0, 0], "k-", lw=2)

    cart = plt.Rectangle((0, 0), cart_w, cart_h, color="tab:blue", ec="k")
    ax.add_patch(cart)
    (pole_line,) = ax.plot([], [], "o-", color="tab:orange", lw=2, markersize=6)

    def _set_frame(i):
        xc, yc = x[i], 0.0
        cart.set_xy((xc - cart_w/2, yc - cart_h/2))
        xp = xc + pole_len * np.sin(th[i])
        yp = yc + pole_len * np.cos(th[i])
        pole_line.set_data([xc, xp], [yc, yp])
        return cart, pole_line

    def init():
        # draw first frame so the figure isn't empty
        return _set_frame(0)

    interval_ms = float((t[1] - t[0]) * 1000) if len(t) > 1 else 20.0
    ani = animation.FuncAnimation(
        fig, _set_frame, frames=len(x), init_func=init,
        blit=True, interval=interval_ms, repeat=False
    )
    # keep a reference so it doesn't get GC'ed
    return fig, ani

