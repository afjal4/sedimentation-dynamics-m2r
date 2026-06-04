import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def render(sim, states, interval=50):
    all_pos = np.array(states)  # (frames, n, 3)
    margin = 0.5
    lims = [(all_pos[:, :, i].min() - margin, all_pos[:, :, i].max() + margin) for i in range(3)]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    ax.set_zlim(*lims[2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    scatter = ax.scatter(*all_pos[0].T, s=10, alpha=0.7, c='royalblue')
    title = ax.set_title('t = 0.00')

    def _update(frame):
        scatter._offsets3d = tuple(all_pos[frame].T)
        title.set_text(f't = {(frame + 1) * sim.dt:.2f}')
        return scatter, title

    anim = FuncAnimation(fig, _update, frames=len(states), interval=interval, blit=False)
    plt.close(fig)
    return anim
