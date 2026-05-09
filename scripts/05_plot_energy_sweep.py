from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot energy sweep CSV outputs.")
    parser.add_argument("--csv", default=Path("results/energy_sweep/energy_sweep.csv"), type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    return parser.parse_args()


def read_rows(csv_path: Path) -> list[dict[str, float]]:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({key: float(value) for key, value in row.items()})
    return rows


def plot_line(rows, y_key: str, ylabel: str, output_path: Path, baseline_key: str | None = None):
    weights = [row["energy_weight"] for row in rows]
    values = [row[y_key] for row in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(weights, values, marker="o", label=y_key)
    if baseline_key is not None:
        baseline = [row[baseline_key] for row in rows]
        plt.plot(weights, baseline, marker="s", label=baseline_key)
        plt.legend()
    plt.xlabel("energy_weight")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_pareto(rows, output_path: Path):
    plt.figure(figsize=(7, 4))
    plt.plot(
        [row["ppo_energy"] for row in rows],
        [row["ppo_deviation"] for row in rows],
        marker="o",
    )
    for row in rows:
        plt.annotate(f'{row["energy_weight"]:.2g}', (row["ppo_energy"], row["ppo_deviation"]))
    plt.xlabel("PPO action energy")
    plt.ylabel("PPO adoption deviation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    args = parse_args()
    rows = read_rows(args.csv)
    if not rows:
        raise ValueError(f"No rows found in {args.csv}")

    output_dir = args.output_dir or args.csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_line(
        rows,
        "ppo_noop_reward_gap",
        "PPO reward - no-op reward",
        output_dir / "ppo_noop_reward_gap.png",
    )
    plot_line(rows, "ppo_energy", "PPO action energy", output_dir / "ppo_action_energy.png")
    plot_line(
        rows,
        "ppo_deviation",
        "Adoption deviation",
        output_dir / "adoption_deviation.png",
        baseline_key="none_deviation",
    )
    plot_pareto(rows, output_dir / "pareto_deviation_vs_energy.png")
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
