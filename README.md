# Platform-Controlled Trend Propagation

This repository contains a Network Science capstone project on platform-controlled trend propagation in social networks. The project models multi-trend competition on the SNAP Facebook ego network using SIS-style contagion dynamics with node-level fatigue, finite attention, and reinforcement-learning control.

## Core Idea

The platform agent tries to keep a target trend near a desired network-wide attention share. Instead of selecting nodes directly with a 4039-dimensional binary action, PPO outputs a compact vector that is decoded into node intervention scores. Several decoders compare spectral graph features, centrality features, and dynamic frontier features.

The main finding is that dynamic frontier information is a much stronger intervention signal than static spectral or centrality information for this fully observed node-level control task. A signed-frontier PPO policy, which can both boost and suppress through the frontier, learns target-dependent regulation and substantially reduces deviation from target share.

## Repository Layout

- `src/dynamics.py`: single-trend and multi-trend SIS-with-fatigue dynamics.
- `src/env.py`: Gymnasium environments for single-trend and multi-trend control.
- `src/graph.py`: graph loading, archetype sampling, affinity generation, spectral features, and centrality features.
- `scripts/01_test_dynamics.py`: validate single-trend and multi-trend dynamics.
- `scripts/02_train_agent.py`: train/evaluate PPO policies.
- `scripts/04_energy_sweep.py`: single-trend energy penalty sweep.
- `scripts/07_decoder_ablation.py`: multi-trend decoder ablation.
- `scripts/11_paper_metrics.py`: reproduce paper-facing metrics and CSVs.
- `scripts/12_ppo_target_sweep.py`: signed-frontier PPO target sweep.
- `scripts/13_signed_frontier_significance.py`: paired statistical tests.
- `scripts/14_visualize_signed_frontier.py`: line plots comparing signed-frontier PPO to baselines.
- `scripts/15_visualize_signed_frontier_graph.py`: graph heatmap visualization for the presentation.
- `results/`: selected paper-facing figures and CSV summaries.

## Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate rl_env
```

If you already have the environment, update it:

```bash
conda env update -f environment.yml --prune
```

## Data

The project expects the SNAP Facebook combined ego network at:

```text
data/facebook_combined.txt
```

The included scripts use this file by default.

## Common Commands

Validate dynamics:

```bash
python scripts/01_test_dynamics.py
python scripts/01_test_dynamics.py --multi-trend --trend-strengths 1.0 1.25 1.0 0.75
```

Train signed-frontier PPO across target regimes:

```bash
python scripts/12_ppo_target_sweep.py \
  --output-dir results/ppo_signed_frontier_target_sweep \
  --timesteps 150000
```

Run statistical tests:

```bash
python scripts/13_signed_frontier_significance.py
```

Generate the presentation graph heatmap:

```bash
python scripts/15_visualize_signed_frontier_graph.py
```

## Key Results

Signed-frontier PPO reduced mean deviation from no intervention by:

- 86.2% at target share 0.20
- 92.8% at target share 0.30
- 60.4% at target share 0.50

All paired tests against no intervention were significant at `p <= 2.1e-7`.

## Notes

Training checkpoints and logs are intentionally ignored by git. The committed CSV files and figures are the lightweight paper-facing artifacts; PPO models can be regenerated from the scripts.
