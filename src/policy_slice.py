import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from .discretizer import Q_SHAPE


ACTION_LABELS: Dict[int, str] = {
    0: "FL",
    1: "FS",
    2: "FR",
    3: "SL",
    4: "SS",
    5: "SR",
    6: "BR",
}


def greedy_policy_slice(
    q: np.ndarray,
    v_bin: int,
    tile: int,
    stop_bin: int,
    sigma_stop: int,
    duck: int,
) -> str:
    expected = Q_SHAPE
    if q.shape != expected:
        raise ValueError(f"Expected Q shape {expected}, got {q.shape}")
    values = q[:, :, v_bin, tile, stop_bin, sigma_stop, duck, :]
    actions = np.argmax(values, axis=-1)
    lines = [
        "Greedy policy slice",
        f"v_bin={v_bin} tile={tile} stop_bin={stop_bin} sigma={sigma_stop} duck={duck}",
        "rows: phi high -> low; columns: d low -> high",
        "      d0 d1 d2 d3 d4",
    ]
    for phi in reversed(range(actions.shape[1])):
        row = " ".join(ACTION_LABELS[int(actions[d, phi])] for d in range(actions.shape[0]))
        lines.append(f"phi{phi}: {row}")
    lines.append("legend: FL/FS/FR fast, SL/SS/SR slow, BR brake")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a 5x5 greedy policy slice over (d, phi).")
    parser.add_argument("--q-table", type=Path, default=Path("results/q_table.npy"))
    parser.add_argument("--v-bin", type=int, choices=range(3), default=2)
    parser.add_argument("--tile", type=int, choices=range(3), default=0)
    parser.add_argument("--stop-bin", type=int, choices=range(4), default=0)
    parser.add_argument("--sigma-stop", type=int, choices=range(2), default=0)
    parser.add_argument("--duck", type=int, choices=range(5), default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = greedy_policy_slice(
        np.load(args.q_table),
        args.v_bin,
        args.tile,
        args.stop_bin,
        args.sigma_stop,
        args.duck,
    )
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

