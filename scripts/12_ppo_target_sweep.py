from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


TARGETS = (0.2, 0.3, 0.5)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO across target-share regimes.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--targets", nargs="+", default=list(TARGETS), type=float)
    parser.add_argument("--output-dir", default=ROOT / "results/ppo_target_sweep", type=Path)
    parser.add_argument("--timesteps", default=150_000, type=int)
    parser.add_argument("--episodes", default=10, type=int)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run_training(args, target: float):
    target_label = f"t{target:.2f}".replace(".", "p")
    run_dir = args.output_dir / target_label
    model_path = run_dir / "ppo_contagion.zip"
    stdout_path = run_dir / "train_eval.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and model_path.exists() and stdout_path.exists():
        return stdout_path

    command = [
        args.python,
        str(ROOT / "scripts/02_train_agent.py"),
        "--multi-trend",
        "--decoder-mode",
        "signed_frontier",
        "--target-trend",
        "3",
        "--target-share",
        str(target),
        "--trend-strengths",
        "1.0",
        "1.25",
        "1.0",
        "0.75",
        "--timesteps",
        str(args.timesteps),
        "--episodes",
        str(args.episodes),
        "--max-steps",
        "100",
        "--spectral-modes",
        "5",
        "--boost-budget",
        "1200",
        "--suppress-budget",
        "1200",
        "--action-scale",
        "2.0",
        "--energy-weight",
        "0.02",
        "--device",
        "cpu",
        "--output-dir",
        str(run_dir),
    ]
    print("Running:", " ".join(command), flush=True)
    with stdout_path.open("w") as handle:
        subprocess.run(command, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)
    return stdout_path


def parse_metrics(log_path: Path):
    rows = []
    for line in log_path.read_text().splitlines():
        if ": {'mean_reward'" not in line:
            continue
        policy, payload = line.split(": ", 1)
        metrics = eval(payload, {"__builtins__": {}}, {})
        row = {"policy": policy}
        row.update(metrics)
        rows.append(row)
    return rows


def write_summary(output_dir: Path, rows: list[dict]):
    if not rows:
        return
    path = output_dir / "ppo_target_sweep.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for target in args.targets:
        log_path = run_training(args, target)
        for row in parse_metrics(log_path):
            row = {"target_share": target, **row}
            all_rows.append(row)
            print(row)
    write_summary(args.output_dir, all_rows)

    metrics_script = ROOT / "scripts/11_paper_metrics.py"
    metrics_out = args.output_dir / "paper_metrics_snapshot"
    if metrics_script.exists():
        metrics_out.mkdir(parents=True, exist_ok=True)
        source = ROOT / "results/paper_metrics/target_regime_sweep.csv"
        if source.exists():
            shutil.copy2(source, metrics_out / source.name)


if __name__ == "__main__":
    main()
