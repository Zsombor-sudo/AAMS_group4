from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from irsim.env import EnvBase
from irsim.world.object_base import ObjectBase
from irsim.world.world import World

# ---------------------------------------------------------------------------
# Grid drawing helper
# ---------------------------------------------------------------------------

def draw_grid(env: EnvBase, cell_size: float, color: str = "lightgray"):
    ax = plt.gca()
    w: World = env._world
    x0, x1 = w.x_range
    y0, y1 = w.y_range

    # Boundaries shifted so integer coordinates are centers (± 0.5 for cell_size=1)
    xs = np.arange(x0 - cell_size/2, x1+1 + cell_size/2, cell_size)
    ys = np.arange(y0 - cell_size/2, y1+1 + cell_size/2, cell_size)

    for x in xs:
        ax.plot([x, x], [ys[0], ys[-1]], color=color, linewidth=0.6, zorder=0)
    for y in ys:
        ax.plot([xs[0], xs[-1]], [y, y], color=color, linewidth=0.6, zorder=0)

    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])

    xticks = np.arange(x0, x1+1, 1)
    yticks = np.arange(y0, y1+1, 1)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlabel("")
    ax.set_ylabel("")


# ---------------------------------------------------------------------------
# Level label helpers (agents & apples)
# ---------------------------------------------------------------------------

def _obj_xy(obj: ObjectBase) -> Tuple[float, float]:
    """Return (x,y) float coordinates for an object (center)."""
    return float(obj.state[0,0]), float(obj.state[1,0])

def init_labels(ax: plt.Axes, agents: list[ObjectBase], apples: list[ObjectBase]) -> tuple[Dict[int, plt.Text], Dict[int, plt.Text]]:
    """Create matplotlib Text artists for agent and apple levels.

    Returns:
        (agent_texts, apple_texts) mapping ids -> Text
    """
    agent_texts: Dict[int, plt.Text] = {}
    apple_texts: Dict[int, plt.Text] = {}

    for a in agents:
        x, y = _obj_xy(a)
        agent_texts[a.id] = ax.text(
            x, y, f"L{getattr(a, 'level', '?')}",
            ha="center", va="center",
            color="white", fontsize=8, weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.6)
        )
    for ap in apples:
        x, y = _obj_xy(ap)
        apple_texts[ap.id] = ax.text(
            x, y, f"L{getattr(ap, 'level', '?')}",
            ha="center", va="center",
            color="yellow", fontsize=8, weight="bold",
            bbox=dict(boxstyle="circle,pad=0.3", fc="green", ec="none", alpha=0.5)
        )
    return agent_texts, apple_texts

def update_labels(ax: plt.Axes, agent_texts: Dict[int, plt.Text], apple_texts: Dict[int, plt.Text],
                        agents: list[ObjectBase], apples: list[ObjectBase]) -> None:
    """Update existing Text artists' positions & visibility based on current states."""
    # Agents
    for a in agents:
        t = agent_texts.get(a.id)
        if not t:
            continue
        x, y = _obj_xy(a)
        t.set_position((x, y))
        t.set_text(f"L{getattr(a, 'level', '?')}")

    # Apples
    for ap in apples:
        t = apple_texts.get(ap.id)
        if ap.__dict__.get('collected', False):  # hide collected
            if t:
                t.set_visible(False)
            continue

        # Dynamic spawning support
        if t is None:
            x, y = _obj_xy(ap)
            apple_texts[ap.id] = ax.text(
                x, y, f"L{getattr(ap, 'level', '?')}",
                ha="center", va="center",
                color="yellow", fontsize=8, weight="bold",
                bbox=dict(boxstyle="circle,pad=0.3", fc="green", ec="none", alpha=0.5)
            )
        else:
            x, y = _obj_xy(ap)
            t.set_visible(True)
            t.set_position((x, y))
            t.set_text(f"L{getattr(ap, 'level', '?')}")