import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap

from env import GridWorld
from dqn import DQN
from drqn import DRQN

# ─── Constants ───────────────────────────────────────────────────────────────
DEVICE    = torch.device("cpu")
GRID_SIZE = 8
MAX_STEPS = 50
SEQ_LEN   = 16
PAUSE_SEC = 0.25

TRAIL_DQN  = LinearSegmentedColormap.from_list("dqn_trail",  ["#0d1b2a", "#00b4d8", "#90e0ef"])
TRAIL_DRQN = LinearSegmentedColormap.from_list("drqn_trail", ["#1a0a2e", "#ff6b6b", "#ffd166"])

BG_DQN   = "#050d1a"
BG_DRQN  = "#110a1a"
ACC_DQN  = "#00b4d8"
ACC_DRQN = "#ff6b6b"


# ─── Model loading ───────────────────────────────────────────────────────────
def _best_seed_model(prefix: str, n_seeds: int = 16):
    best_seed, best_score = None, -1e9
    for seed in range(n_seeds):
        try:
            rewards = np.load(f"{prefix}_rewards_seed{seed}.npy")
            score   = rewards[-50:].mean()
            if score > best_score:
                best_score, best_seed = score, seed
        except FileNotFoundError:
            continue
    return best_seed, best_score


def load_dqn() -> DQN:
    model = DQN(input_size=25, actions=4)
    seed, score = _best_seed_model("dqn")
    if seed is not None:
        model.load_state_dict(torch.load(f"dqn_model_seed{seed}.pth", map_location=DEVICE))
        print(f"Loaded BEST DQN  -- seed {seed}  (score {score:.2f})")
    else:
        print("Warning: no DQN weights found")
    return model.eval()


def load_drqn() -> DRQN:
    model = DRQN(input_size=25, hidden=64, actions=4)
    seed, score = _best_seed_model("drqn")
    if seed is not None:
        model.load_state_dict(torch.load(f"drqn_model_seed{seed}.pth", map_location=DEVICE))
        print(f"Loaded BEST DRQN -- seed {seed}  (score {score:.2f})")
    else:
        print("Warning: no DRQN weights found")
    return model.eval()


# ─── Drawing helpers ─────────────────────────────────────────────────────────
def draw_mini_robot(ax, position, angle_deg, color_body="#00b4d8", color_accent="#ffffff"):
    cx, cy = position[1], position[0]
    angle  = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    def rot(dx, dy):
        return cx + cos_a * dx - sin_a * dy, cy - sin_a * dx - cos_a * dy

    ax.add_patch(FancyBboxPatch(
        (cx - 0.22, cy - 0.22), 0.44, 0.44,
        boxstyle="round,pad=0.06", linewidth=1.8,
        edgecolor=color_accent, facecolor=color_body, zorder=6,
    ))
    ax.add_patch(FancyBboxPatch(
        (cx - 0.17, cy - 0.06), 0.34, 0.10,
        boxstyle="round,pad=0.02", linewidth=0,
        facecolor=color_accent, alpha=0.25, zorder=7,
    ))
    for ex_off in (-0.09, 0.09):
        ex, ey = rot(ex_off, -0.06)
        ax.scatter(ex, ey, s=55, color=color_accent, alpha=0.3, zorder=7)
        ax.scatter(ex, ey, s=18, color=color_accent, zorder=8)
    nx, ny = rot(0, -0.28)
    ax.plot([cx, nx], [cy, ny], color=color_accent, lw=2.0,
            solid_capstyle="round", zorder=8)
    ax.scatter(nx, ny, s=14, color=color_accent, zorder=9)
    for side in (-0.10, 0.10):
        ax_tip,  ay_tip  = rot(side,       -0.38)
        ax_base, ay_base = rot(side * 0.6, -0.22)
        ax.plot([ax_base, ax_tip], [ay_base, ay_tip],
                color=color_accent, lw=1.2, alpha=0.8, zorder=7)
        ax.scatter(ax_tip, ay_tip, s=10, color=color_accent, alpha=0.9, zorder=8)
    for side in (-0.16, 0.16):
        wx, wy = rot(side, 0.24)
        ax.add_patch(Circle((wx, wy), 0.07,
                             color="#1a1a2e", ec=color_body, lw=1.2, zorder=7))


def draw_goal(ax, goal_pos):
    gx, gy = goal_pos[1], goal_pos[0]
    for r, a in ((0.42, 0.08), (0.30, 0.18), (0.18, 0.35)):
        ax.add_patch(Circle((gx, gy), r, color="#00ff88", alpha=a, zorder=3, linewidth=0))
    ax.add_patch(Circle((gx, gy), 0.14, color="#00ff88",
                         fill=False, lw=2.5, alpha=0.9, zorder=4))
    ax.scatter(gx, gy, s=120, color="#00ff88", zorder=5,
               path_effects=[pe.withSimplePatchShadow(
                   offset=(0, 0), shadow_rgbFace="#00ff88", alpha=0.5, rho=3.0)])
    for ang in range(0, 360, 45):
        sx = gx + 0.22 * np.cos(np.radians(ang))
        sy = gy + 0.22 * np.sin(np.radians(ang))
        ax.plot([gx, sx], [gy, sy], color="#00ff88", lw=0.8, alpha=0.55, zorder=4)
    ax.text(gx, gy - 0.38, "GOAL", ha="center", va="center", fontsize=5.5,
            color="#00ff88", fontweight="bold", fontfamily="monospace",
            alpha=0.85, zorder=6)


def draw_obstacles(ax, obstacles, accent_color):
    """
    Draw each obstacle as a compact, clean wall block:
      - solid dark fill
      - 3 thin diagonal hazard lines (not overflowing the cell)
      - thin glowing border
    Kept tight inside the cell so it doesn't overwhelm the grid.
    """
    PAD = 0.40   # half-width of the block (< 0.5 so there's a gap between cells)

    for (rr, cc) in obstacles:
        cx, cy = float(cc), float(rr)
        x0, y0 = cx - PAD, cy - PAD
        w = h = PAD * 2

        # ── solid dark base ───────────────────────────────────────────────
        ax.add_patch(plt.Rectangle(
            (x0, y0), w, h,
            color="#0d0d20", alpha=0.95, zorder=2,
        ))

        # ── 3 thin diagonal hazard lines clipped inside the cell ──────────
        stripe_gap = w / 3.5
        stripe_col = "#e8b000"
        for k in range(-1, 4):
            sx = x0 + k * stripe_gap
            # line from (sx, y0) to (sx + h, y0 + h) — 45-degree
            lx = [sx, sx + h]
            ly = [y0, y0 + h]
            # clip to [x0, x0+w]
            pts = []
            for i in range(len(lx) - 1):
                x1_, y1_ = lx[i],   ly[i]
                x2_, y2_ = lx[i+1], ly[i+1]
                # simple horizontal clip
                if x2_ < x0 or x1_ > x0 + w:
                    continue
                x1_ = max(x1_, x0);  x2_ = min(x2_, x0 + w)
                dx = lx[i+1] - lx[i]
                if abs(dx) < 1e-9:
                    continue
                t1 = (x1_ - lx[i]) / dx
                t2 = (x2_ - lx[i]) / dx
                cy1 = ly[i] + t1 * (ly[i+1] - ly[i])
                cy2 = ly[i] + t2 * (ly[i+1] - ly[i])
                # vertical clip
                cy1c = np.clip(cy1, y0, y0 + h)
                cy2c = np.clip(cy2, y0, y0 + h)
                pts.append(((x1_, cy1c), (x2_, cy2c)))
            for (px1, py1), (px2, py2) in pts:
                ax.plot([px1, px2], [py1, py2],
                        color=stripe_col, lw=1.4, alpha=0.65,
                        solid_capstyle="butt", zorder=3)

        # ── glowing border ────────────────────────────────────────────────
        ax.add_patch(plt.Rectangle(
            (x0, y0), w, h,
            fill=False, edgecolor=accent_color,
            linewidth=1.2, alpha=0.55, zorder=4,
        ))

        # ── small "X" in centre ───────────────────────────────────────────
        s = 0.13
        for dx_, dy_, ex_, ey_ in [
            (-s, -s, s, s),
            ( s, -s, -s, s),
        ]:
            ax.plot([cx + dx_, cx + ex_], [cy + dy_, cy + ey_],
                    color=accent_color, lw=1.0, alpha=0.50, zorder=5)


def draw_trail(ax, path, visited, cmap):
    if len(path) < 2:
        return
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    pts  = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    t    = np.linspace(0, 1, len(segs))
    lc   = LineCollection(segs, cmap=cmap, alpha=0.55, linewidth=2.5, zorder=5)
    lc.set_array(t)
    ax.add_collection(lc)
    intens = np.array([visited[r_, c_] for c_, r_ in path], dtype=float)
    intens /= visited.max() + 1e-5
    ax.scatter(xs, ys, c=intens, cmap=cmap, s=22, alpha=0.55, zorder=5)


def style_ax(ax, title, bg_color, accent_color):
    ax.set_facecolor(bg_color)
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(GRID_SIZE - 0.5, -0.5)
    for i in range(GRID_SIZE + 1):
        ax.axhline(i - 0.5, color=accent_color, lw=0.4, alpha=0.20, zorder=1)
        ax.axvline(i - 0.5, color=accent_color, lw=0.4, alpha=0.20, zorder=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(accent_color)
        spine.set_linewidth(1.5)
        spine.set_alpha(0.5)
    ax.set_title(title, fontsize=10, color=accent_color,
                 fontfamily="monospace", fontweight="bold", pad=8)


def draw_status_badge(ax, done: bool, timed_out: bool, accent: str):
    """Green DONE / orange TIMEOUT / accent LIVE — no emoji (avoids font warnings)."""
    if done:
        label, color = "[ DONE ]", "#00ff88"
    elif timed_out:
        label, color = "[ TIMEOUT ]", "#ff9f43"
    else:
        label, color = "[ LIVE ]", accent
    ax.text(0.97, 0.97, label, transform=ax.transAxes,
            fontsize=8, ha="right", va="top",
            color=color, fontfamily="monospace", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a1a",
                      edgecolor=color, linewidth=1.0, alpha=0.90),
            zorder=10)


def draw_step_counter(ax, step, max_steps, accent):
    """Progress bar at the bottom of each grid panel."""
    frac = step / max_steps
    # background bar
    ax.add_patch(plt.Rectangle(
        (-0.5, GRID_SIZE - 0.5 + 0.05), GRID_SIZE, 0.18,
        color="#1a1a2e", alpha=0.7, zorder=9,
        clip_on=False,
    ))
    # filled portion
    ax.add_patch(plt.Rectangle(
        (-0.5, GRID_SIZE - 0.5 + 0.05), GRID_SIZE * frac, 0.18,
        color=accent, alpha=0.55, zorder=10,
        clip_on=False,
    ))


def manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ─── Agent step helpers ──────────────────────────────────────────────────────
def step_dqn(model, state, env, prev_pos, eps: float = 0.05):
    st = torch.FloatTensor(state.flatten()).unsqueeze(0)
    # FIX 2: small epsilon-greedy nudge so agent escapes local loops during demo
    if np.random.rand() < eps:
        action = np.random.randint(4)
    else:
        action = torch.argmax(model(st)).item()
    next_state, _, done = env.step(action)
    r, c = env.agent_pos
    dx = c - (prev_pos[1] if prev_pos else c)
    dy = (prev_pos[0] if prev_pos else r) - r
    angle = np.degrees(np.arctan2(dy, dx)) if (dx or dy) else 90
    return next_state, done, (r, c), angle


def step_drqn(model, state_seq, state, env, hidden, prev_pos):
    state_seq.append(state.flatten())
    if len(state_seq) > SEQ_LEN:
        state_seq = state_seq[-SEQ_LEN:]
    pad     = [np.zeros_like(state_seq[0])] * max(0, SEQ_LEN - len(state_seq))
    seq_arr = np.array(pad + state_seq, dtype=np.float32)
    seq_t   = torch.FloatTensor(seq_arr).unsqueeze(0)
    q, hidden = model(seq_t, hidden)
    # FIX 2: small epsilon-greedy nudge (same eps as DQN for fair comparison)
    if np.random.rand() < 0.05:
        action = np.random.randint(4)
    else:
        action = torch.argmax(q).item()
    next_state, _, done = env.step(action)
    r, c = env.agent_pos
    dx = c - (prev_pos[1] if prev_pos else c)
    dy = (prev_pos[0] if prev_pos else r) - r
    angle = np.degrees(np.arctan2(dy, dx)) if (dx or dy) else 90
    return next_state, done, (r, c), angle, hidden, state_seq


# ─── Main animation ──────────────────────────────────────────────────────────
def animate_side_by_side():

    # FIX 1: regenerate until map is solvable and not overly dense.
    # reset() already calls _is_reachable internally, but high obstacle_prob can
    # still produce tight mazes where the greedy agent loops forever. Cap at 14
    # obstacles and do an explicit reachability check as a second safety net.
    MAX_OBS = 14
    while True:
        env_dqn    = GridWorld(grid_size=GRID_SIZE, dynamic_obstacles=True, obstacle_prob=0.18)
        init_state = env_dqn.reset()
        if (len(env_dqn.obstacles) <= MAX_OBS
                and env_dqn._is_reachable(env_dqn.agent_pos, env_dqn.goal_pos)):
            break
    env_dqn.dynamic_obstacles = False   # freeze map

    # env_drqn: identical map, same start, same goal
    env_drqn = GridWorld(
        grid_size=GRID_SIZE,
        obstacles=list(env_dqn.obstacles),
        dynamic_obstacles=False,
    )
    env_drqn.agent_pos = env_dqn.agent_pos
    env_drqn.goal_pos  = env_dqn.goal_pos

    print(f"Obstacles ({len(env_dqn.obstacles)}): {sorted(env_dqn.obstacles)}")
    print(f"Start : {env_dqn.agent_pos}   Goal : {env_dqn.goal_pos}")
    assert env_dqn.agent_pos == env_drqn.agent_pos, "Start mismatch!"
    assert env_dqn.goal_pos  == env_drqn.goal_pos,  "Goal mismatch!"
    assert sorted(env_dqn.obstacles) == sorted(env_drqn.obstacles), "Obstacle mismatch!"

    state_dqn  = init_state.copy()
    state_drqn = init_state.copy()

    dqn_agent  = load_dqn()
    drqn_agent = load_drqn()

    hidden       = None
    state_seq    = []
    hidden_norms = []

    visited_dqn  = np.zeros((GRID_SIZE, GRID_SIZE))
    visited_drqn = np.zeros((GRID_SIZE, GRID_SIZE))
    path_dqn     = []
    path_drqn    = []

    prev_dqn  = None
    prev_drqn = None
    done_dqn  = False
    done_drqn = False

    # ── figure ───────────────────────────────────────────────────────────────
    plt.ion()
    fig = plt.figure(figsize=(14, 8), facecolor="#07071a")
    gs  = fig.add_gridspec(2, 2, height_ratios=[3.2, 1],
                           hspace=0.30, wspace=0.06,
                           left=0.04, right=0.97, top=0.92, bottom=0.08)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    fig.suptitle("DQN  vs  DRQN  --  Grid Navigation",
                 fontsize=14, color="#e0e0ff",
                 fontfamily="monospace", fontweight="bold")

    # ── step loop ────────────────────────────────────────────────────────────
    for step in range(MAX_STEPS):

        if not done_dqn:
            state_dqn, done_dqn, pos_dqn, angle_dqn = step_dqn(
                dqn_agent, state_dqn, env_dqn, prev_dqn)
            visited_dqn[pos_dqn] += 1
            path_dqn.append((pos_dqn[1], pos_dqn[0]))
            prev_dqn = pos_dqn
        else:
            angle_dqn = 90

        if not done_drqn:
            state_drqn, done_drqn, pos_drqn, angle_drqn, hidden, state_seq = step_drqn(
                drqn_agent, state_seq, state_drqn, env_drqn, hidden, prev_drqn)
            visited_drqn[pos_drqn] += 1
            path_drqn.append((pos_drqn[1], pos_drqn[0]))
            prev_drqn = pos_drqn
            if hidden is not None:
                hidden_norms.append(torch.norm(hidden[0]).item())
            # FIX 3: always clear LSTM hidden state when episode ends to prevent
            # memory from a finished episode bleeding into a new one
            if done_drqn:
                hidden     = None
                state_seq  = []   # also clear the sequence buffer
        else:
            angle_drqn = 90

        timed_out = (step == MAX_STEPS - 1)

        print(f"Step {step+1:02d} | DQN  pos={env_dqn.agent_pos}  done={done_dqn} "
              f"| DRQN pos={env_drqn.agent_pos}  done={done_drqn}")

        # ── redraw ───────────────────────────────────────────────────────────
        ax1.clear()
        ax2.clear()

        h_str = f"{hidden_norms[-1]:.2f}" if hidden_norms else "--"
        style_ax(ax1, f"[ DQN -- No Memory ]  step {step+1:02d}",   BG_DQN,  ACC_DQN)
        style_ax(ax2, f"[ DRQN -- h={h_str} ]  step {step+1:02d}", BG_DRQN, ACC_DRQN)

        # obstacles (compact blocks with hazard stripes)
        draw_obstacles(ax1, env_dqn.obstacles,  ACC_DQN)
        draw_obstacles(ax2, env_drqn.obstacles, ACC_DRQN)

        # agent trails
        draw_trail(ax1, path_dqn,  visited_dqn,  TRAIL_DQN)
        draw_trail(ax2, path_drqn, visited_drqn, TRAIL_DRQN)

        # goal markers
        draw_goal(ax1, env_dqn.goal_pos)
        draw_goal(ax2, env_drqn.goal_pos)

        # robots
        draw_mini_robot(ax1, env_dqn.agent_pos,  angle_dqn,
                        color_body=ACC_DQN,  color_accent="#e0f7ff")
        draw_mini_robot(ax2, env_drqn.agent_pos, angle_drqn,
                        color_body=ACC_DRQN, color_accent="#fff0f0")

        # status badge (no emoji — avoids font warnings)
        draw_status_badge(ax1, done_dqn,  timed_out and not done_dqn,  ACC_DQN)
        draw_status_badge(ax2, done_drqn, timed_out and not done_drqn, ACC_DRQN)

        # step progress bar
        draw_step_counter(ax1, step + 1, MAX_STEPS, ACC_DQN)
        draw_step_counter(ax2, step + 1, MAX_STEPS, ACC_DRQN)

        # ── LSTM sparkline ────────────────────────────────────────────────────
        ax3.clear()
        ax3.set_facecolor("#07070b")
        if hidden_norms:
            xs = list(range(len(hidden_norms)))
            ax3.fill_between(xs, hidden_norms, alpha=0.15, color="#ffd166")
            ax3.plot(hidden_norms, color="#ffd166", linewidth=1.8)
            ax3.scatter([xs[-1]], [hidden_norms[-1]], color="#ffd166", s=45, zorder=5)
        ax3.set_xlim(0, max(MAX_STEPS, len(hidden_norms) + 1))
        ax3.set_ylim(0, max(1.0, (max(hidden_norms) if hidden_norms else 1.0) * 1.15))
        ax3.set_xlabel("Step",             color="#aaaaaa", fontsize=8)
        ax3.set_ylabel("LSTM hidden norm", color="#aaaaaa", fontsize=8)
        ax3.tick_params(colors="#888888", labelsize=7)
        ax3.grid(alpha=0.10)
        ax3.set_title("LSTM memory activity  (hidden-state norm)",
                      color="#ffd166", fontsize=9, fontfamily="monospace")

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(PAUSE_SEC)

        if done_dqn and done_drqn:
            break

    # ── final result overlay ──────────────────────────────────────────────────
    dist_dqn  = manhattan(tuple(env_dqn.agent_pos),  tuple(env_dqn.goal_pos))
    dist_drqn = manhattan(tuple(env_drqn.agent_pos), tuple(env_drqn.goal_pos))

    if dist_dqn < dist_drqn:
        winner_dqn, winner_drqn = " << WIN", ""
    elif dist_drqn < dist_dqn:
        winner_dqn, winner_drqn = "", " << WIN"
    else:
        winner_dqn = winner_drqn = " TIE"

    steps_dqn  = len(path_dqn)
    steps_drqn = len(path_drqn)

    for ax, dist, steps, col, tag in (
        (ax1, dist_dqn,  steps_dqn,  ACC_DQN,  winner_dqn),
        (ax2, dist_drqn, steps_drqn, ACC_DRQN, winner_drqn),
    ):
        result = "REACHED GOAL" if dist == 0 else f"dist={dist}"
        ax.text(
            0.5, 0.05,
            f"  {result}  |  {steps} steps{tag}  ",
            transform=ax.transAxes,
            fontsize=9, ha="center", va="bottom",
            color=col, fontfamily="monospace", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#07071a",
                      edgecolor=col, linewidth=1.4, alpha=0.92),
            zorder=10,
        )

    fig.canvas.draw()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    animate_side_by_side()