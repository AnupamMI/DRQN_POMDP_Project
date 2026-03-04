import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Wedge
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from env import GridWorld
from dqn import DQN
from drqn import DRQN

DEVICE = torch.device("cpu")
GRID_SIZE = 5
MAX_STEPS = 50

# ─── Custom colormaps ───────────────────────────────────────────────────────
TRAIL_DQN  = LinearSegmentedColormap.from_list("dqn_trail",  ["#0d1b2a", "#00b4d8", "#90e0ef"])
TRAIL_DRQN = LinearSegmentedColormap.from_list("drqn_trail", ["#1a0a2e", "#ff6b6b", "#ffd166"])


def load_dqn():
    model = DQN(9, 4)
    model.load_state_dict(torch.load("dqn_model_seed1.pth", map_location=DEVICE))
    model.eval()
    return model


def load_drqn():
    model = DRQN(9, 64, 4)
    model.load_state_dict(torch.load("drqn_model_seed1.pth", map_location=DEVICE))
    model.eval()
    return model


# ─── Mini-robot drawing ──────────────────────────────────────────────────────
def draw_mini_robot(ax, position, angle_deg, color_body="#00b4d8", color_accent="#ffffff"):
    """
    Draws a cute mini-robot:
      - rounded square body
      - two antennae
      - two glowing eyes
      - a visor stripe
      - two small wheel bumps at the bottom
    All centred at grid cell (position[1], position[0]).
    """
    cx = position[1]   # column → x
    cy = position[0]   # row    → y

    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    def rot(dx, dy):
        """Rotate offset (dx, dy) by the heading angle."""
        return cx + cos_a * dx - sin_a * dy, cy - sin_a * dx - cos_a * dy  # note: y-axis is flipped in grid

    # ── body ──────────────────────────────────────────────
    body = FancyBboxPatch(
        (cx - 0.22, cy - 0.22), 0.44, 0.44,
        boxstyle="round,pad=0.06",
        linewidth=1.8,
        edgecolor=color_accent,
        facecolor=color_body,
        zorder=6,
    )
    ax.add_patch(body)

    # ── visor stripe ──────────────────────────────────────
    visor = FancyBboxPatch(
        (cx - 0.17, cy - 0.06), 0.34, 0.10,
        boxstyle="round,pad=0.02",
        linewidth=0,
        facecolor=color_accent,
        alpha=0.25,
        zorder=7,
    )
    ax.add_patch(visor)

    # ── eyes ──────────────────────────────────────────────
    for ex_off in [-0.09, 0.09]:
        ex, ey = rot(ex_off, -0.06)
        # glow halo
        ax.scatter(ex, ey, s=55, color=color_accent, alpha=0.3, zorder=7)
        # bright pupil
        ax.scatter(ex, ey, s=18, color=color_accent, zorder=8)

    # ── nose / direction indicator ─────────────────────────
    nx, ny = rot(0, -0.28)
    ax.plot([cx, nx], [cy, ny],
            color=color_accent, lw=2.0, solid_capstyle="round", zorder=8)
    ax.scatter(nx, ny, s=14, color=color_accent, zorder=9)

    # ── antennae ──────────────────────────────────────────
    for side in [-0.10, 0.10]:
        ax_tip, ay_tip = rot(side, -0.38)
        ax_base, ay_base = rot(side * 0.6, -0.22)
        ax.plot([ax_base, ax_tip], [ay_base, ay_tip],
                color=color_accent, lw=1.2, alpha=0.8, zorder=7)
        ax.scatter(ax_tip, ay_tip, s=10, color=color_accent, alpha=0.9, zorder=8)

    # ── wheels ────────────────────────────────────────────
    for side in [-0.16, 0.16]:
        wx, wy = rot(side, 0.24)
        wheel = Circle((wx, wy), 0.07,
                        color="#1a1a2e", ec=color_body, lw=1.2, zorder=7)
        ax.add_patch(wheel)


# ─── Goal-position drawing ───────────────────────────────────────────────────
def draw_goal(ax, goal_pos):
    """
    Draws a glowing, pulsing-style green goal marker:
      - three concentric halos (alpha layers)
      - bright inner dot with a star
    """
    gx = goal_pos[1]
    gy = goal_pos[0]

    # outer halos
    for r, a in [(0.42, 0.08), (0.30, 0.18), (0.18, 0.35)]:
        halo = Circle((gx, gy), r,
                      color="#00ff88", alpha=a, zorder=3, linewidth=0)
        ax.add_patch(halo)

    # ring
    ring = Circle((gx, gy), 0.14,
                  color="#00ff88", fill=False, lw=2.5,
                  alpha=0.9, zorder=4)
    ax.add_patch(ring)

    # centre dot
    ax.scatter(gx, gy, s=120, color="#00ff88", zorder=5,
               path_effects=[pe.withSimplePatchShadow(offset=(0, 0),
                                                       shadow_rgbFace="#00ff88",
                                                       alpha=0.5,
                                                       rho=3.0)])

    # tiny star lines
    for ang in range(0, 360, 45):
        r = 0.22
        sx = gx + r * np.cos(np.radians(ang))
        sy = gy + r * np.sin(np.radians(ang))
        ax.plot([gx, sx], [gy, sy],
                color="#00ff88", lw=0.8, alpha=0.55, zorder=4)

    # label
    ax.text(gx, gy - 0.38, "GOAL",
            ha="center", va="center", fontsize=5.5,
            color="#00ff88", fontweight="bold",
            fontfamily="monospace", alpha=0.85, zorder=6)


# ─── Axis styling ────────────────────────────────────────────────────────────
def style_ax(ax, title, bg_color, accent_color):
    ax.set_facecolor(bg_color)
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(GRID_SIZE - 0.5, -0.5)

    # grid lines
    for i in range(GRID_SIZE + 1):
        ax.axhline(i - 0.5, color=accent_color, lw=0.4, alpha=0.25, zorder=1)
        ax.axvline(i - 0.5, color=accent_color, lw=0.4, alpha=0.25, zorder=1)

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_edgecolor(accent_color)
        spine.set_linewidth(1.5)
        spine.set_alpha(0.5)

    ax.set_title(title, fontsize=11, color=accent_color,
                 fontfamily="monospace", fontweight="bold", pad=8)


# ─── Main animation ──────────────────────────────────────────────────────────
def animate_side_by_side():

    env_dqn  = GridWorld()
    env_drqn = GridWorld()

    state_dqn  = env_dqn.reset()
    state_drqn = env_drqn.reset()

    dqn_agent  = load_dqn()
    drqn_agent = load_drqn()

    hidden     = None
    state_seq  = []

    visited_dqn  = np.zeros((GRID_SIZE, GRID_SIZE))
    visited_drqn = np.zeros((GRID_SIZE, GRID_SIZE))

    path_dqn  = []   # list of (col, row)
    path_drqn = []

    prev_pos_dqn  = None
    prev_pos_drqn = None
    hidden_norms  = []

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 6),
        facecolor="#0a0a1a",
        gridspec_kw={"wspace": 0.08}
    )

    fig.suptitle("DQN  vs  DRQN — Grid Navigation",
                 fontsize=14, color="#e0e0ff",
                 fontfamily="monospace", fontweight="bold", y=1.01)

    # ── step loop ────────────────────────────────────────────────────────────
    for step in range(MAX_STEPS):

        # ── DQN step ──────────────────────────────────────────────────────
        st = torch.FloatTensor(state_dqn.flatten()).unsqueeze(0)
        action_dqn = torch.argmax(dqn_agent(st)).item()
        state_dqn, _, done_dqn = env_dqn.step(action_dqn)

        r, c = env_dqn.agent_pos
        visited_dqn[r, c] += 1
        path_dqn.append((c, r))

        dx = c - (prev_pos_dqn[1] if prev_pos_dqn else c)
        dy = (prev_pos_dqn[0] if prev_pos_dqn else r) - r
        angle_dqn = np.degrees(np.arctan2(dy, dx)) if (dx or dy) else 90
        prev_pos_dqn = (r, c)

        # ── DRQN step ─────────────────────────────────────────────────────
        state_seq.append(state_drqn.flatten())
        if len(state_seq) > 4:
            state_seq.pop(0)

        seq_tensor = torch.FloatTensor(np.array(state_seq)).unsqueeze(0)
        q_drqn, hidden = drqn_agent(seq_tensor, hidden)
        action_drqn = torch.argmax(q_drqn).item()
        state_drqn, _, done_drqn = env_drqn.step(action_drqn)

        r2, c2 = env_drqn.agent_pos
        visited_drqn[r2, c2] += 1
        path_drqn.append((c2, r2))

        if hidden is not None:
            hidden_norms.append(torch.norm(hidden[0]).item())

        dx2 = c2 - (prev_pos_drqn[1] if prev_pos_drqn else c2)
        dy2 = (prev_pos_drqn[0] if prev_pos_drqn else r2) - r2
        angle_drqn = np.degrees(np.arctan2(dy2, dx2)) if (dx2 or dy2) else 90
        prev_pos_drqn = (r2, c2)

        # ── draw ─────────────────────────────────────────────────────────
        ax1.clear()
        ax2.clear()

        style_ax(ax1, f"[ DQN — No Memory ]  step {step+1:02d}",
                 bg_color="#050d1a", accent_color="#00b4d8")
        style_ax(ax2,
                 f"[ DRQN — Memory ‖ h={hidden_norms[-1]:.2f} ]  step {step+1:02d}"
                 if hidden_norms else f"[ DRQN — With Memory ]  step {step+1:02d}",
                 bg_color="#110a1a", accent_color="#ff6b6b")

        # heat-trails
        for ax, path, visited, cmap in [
            (ax1, path_dqn,  visited_dqn,  TRAIL_DQN),
            (ax2, path_drqn, visited_drqn, TRAIL_DRQN),
        ]:
            if len(path) > 1:
                xs = [p[0] for p in path]
                ys = [p[1] for p in path]
                # coloured line segments
                pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                t_vals = np.linspace(0, 1, len(segs))
                lc = LineCollection(segs, cmap=cmap, alpha=0.55,
                                    linewidth=2.5, zorder=2)
                lc.set_array(t_vals)
                ax.add_collection(lc)

            if path:
                intens = np.array([visited[r_, c_] for c_, r_ in path])
                norm_i = intens / (visited.max() + 1e-5)
                ax.scatter([p[0] for p in path],
                           [p[1] for p in path],
                           c=norm_i, cmap=cmap,
                           s=28, alpha=0.6, zorder=3)

        # goals
        draw_goal(ax1, env_dqn.goal_pos)
        draw_goal(ax2, env_drqn.goal_pos)

        # robots
        draw_mini_robot(ax1, env_dqn.agent_pos,  angle_dqn,
                        color_body="#00b4d8", color_accent="#e0f7ff")
        draw_mini_robot(ax2, env_drqn.agent_pos, angle_drqn,
                        color_body="#ff6b6b", color_accent="#fff0f0")

        plt.pause(0.35)

        if done_dqn and done_drqn:
            break

    # ── final distance overlay ────────────────────────────────────────────
    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    dist_dqn  = manhattan(env_dqn.agent_pos,  env_dqn.goal_pos)
    dist_drqn = manhattan(env_drqn.agent_pos, env_drqn.goal_pos)

    for ax, dist, col in [
        (ax1, dist_dqn,  "#00b4d8"),
        (ax2, dist_drqn, "#ff6b6b"),
    ]:
        ax.text(
            0.5, 0.06,
            f"▶  Final distance to goal : {dist}",
            transform=ax.transAxes,
            fontsize=9, ha="center", va="bottom",
            color=col, fontfamily="monospace", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#0a0a1a", edgecolor=col,
                      linewidth=1.2, alpha=0.88),
            zorder=10,
        )

    fig.canvas.draw()
    plt.ioff()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    animate_side_by_side()