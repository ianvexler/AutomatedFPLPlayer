import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import pandas as pd
from utils.chips import Chip
from data.data_loader import DataLoader
from simulation import Simulation
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle
import random
import csv
import statistics
import numpy as np

class SimulationEvaluation:
  def __init__(
    self, 
    season, 
    iterations=100, 
    transfers_strategy='weighted',
    load_saved=False
  ):
    self.season = season
    self.ITERATIONS = iterations
    self.load_saved = load_saved
    self.transfers_strategy = transfers_strategy

    self.data_loader = DataLoader()
    self.gw_data = self.data_loader.get_gw_predictions(season)

    self.labels = ['xP', 'fpl_xP']
    # self.labels = ['xP']

    self.DIRECTORY = f'plots/simulation'
    os.makedirs(self.DIRECTORY, exist_ok=True)

  def _run_simulation(self, target):
    best_strategy = {
      Chip.TRIPLE_CAPTAIN: 'conservative',
      Chip.WILDCARD: 'wait',
      Chip.FREE_HIT: 'double_gw',
      Chip.BENCH_BOOST: 'with_wildcard'
    }

    simulation = Simulation(
      season=self.season,
      chip_strategy=best_strategy,
      target=target,
      transfers_strategy=self.transfers_strategy,
      debug=False
    )

    total_points, sim_histories = simulation.simulate_season()
    return total_points, sim_histories, best_strategy

  def evaluate(self):
    random.seed(42)
    
    result_dirs = [
      Path(f"data/cached/simulation/xP_weighted_{self.season}"),
      Path(f"data/cached/simulation/fpl_xP_weighted_{self.season}")
    ]

    results_list = []

    for idx, result_dir in enumerate(result_dirs):
      result_dir.mkdir(parents=True, exist_ok=True)

      if self.load_saved:
        with open(result_dir / "simulation_evaluation_metrics.pkl", "rb") as f:
          evaluation_metrics = pickle.load(f)
      else:
        target = self.labels[idx]
        
        all_sim_points = []
        all_sim_histories = []
        all_chip_strategies = []
        best_sim_points = 0
        best_sim_histories = None
        best_sim_chip_strategy = None

        for i in range(self.ITERATIONS):
          print(f"[{target}] Iteration {i+1}")
          sim_points, sim_histories, chip_strategy = self._run_simulation(target)
          all_sim_points.append(sim_points)
          all_sim_histories.append(sim_histories)
          all_chip_strategies.append(chip_strategy)

          if sim_points > best_sim_points:
            best_sim_points = sim_points
            best_sim_histories = sim_histories
            best_sim_chip_strategy = chip_strategy

        evaluation_metrics = {
          'best_sim_histories': best_sim_histories,
          'best_sim_points': best_sim_points,
          'best_sim_chip_strategy': best_sim_chip_strategy,
          'all_sim_points': all_sim_points,
          'all_sim_histories': all_sim_histories,
          'all_chip_strategies': all_chip_strategies
        }

        with open(result_dir / f"simulation_evaluation_metrics.pkl", "wb") as f:
          pickle.dump(evaluation_metrics, f)

      results = self._evaluate_simulation(evaluation_metrics)
      results_list.append(results)

    self._plot_comparison_distributions([r['final_points_distribution'] for r in results_list])
    self._plot_comparison_weekly(
      [r['cumulative_avg_points'] for r in results_list],
      [r['cumulative_best_points'] for r in results_list]
    )
    self._plot_comparison_weekly_points(
      [r['points_per_gw'] for r in results_list],
      [r['best_points_per_gw'] for r in results_list]
    )

  def _evaluate_simulation(self, evaluation_metrics):
    histories = evaluation_metrics['all_sim_histories']
    best_points = evaluation_metrics['best_sim_points']
    best_sim_histories = evaluation_metrics['best_sim_histories']
    all_sim_points = evaluation_metrics['all_sim_points']

    points_histories = [h.get("points", {}) for h in histories]
    gw_order = sorted(points_histories[0].keys())

    average_points_per_gw = []
    std_dev_per_gw = []
    cumulative_avg_points = []
    best_points_per_gw = []
    cumulative_best_points = []
    cum_avg = 0
    cum_best = 0

    for gw in gw_order:
      gw_values = [h.get(gw, 0) for h in points_histories]
      avg = sum(gw_values) / len(gw_values)
      std = pd.Series(gw_values).std()
      best_gw_points = best_sim_histories.get("points", {}).get(gw, 0)

      # print(f"GW{gw}: {avg}")
      average_points_per_gw.append(avg)
      std_dev_per_gw.append(std)

      cum_avg += avg
      cum_best += best_gw_points

      cumulative_avg_points.append(cum_avg)
      best_points_per_gw.append(best_gw_points)
      cumulative_best_points.append(cum_best)

    final_totals = [sum(sim.values()) for sim in points_histories]

    summary = {
      'best_sim_points': best_points,
      'avg_sim_points': sum(all_sim_points) / len(all_sim_points),
      'stddev_sim_points': statistics.stdev([float(x) for x in all_sim_points]) if len(all_sim_points) > 1 else 0
    }

    print(summary)

    return {
      'summary': summary,
      'points_per_gw': average_points_per_gw,
      'best_points_per_gw': best_points_per_gw,
      'cumulative_avg_points': cumulative_avg_points,
      'cumulative_best_points': cumulative_best_points,
      'stddev_per_gw': std_dev_per_gw,
      'final_points_distribution': final_totals,
      'raw_points': points_histories
    }

  def _plot_comparison_distributions(self, all_distributions, bins=20):
    plt.figure(figsize=(10, 6))
  
    for i, dist in enumerate(all_distributions):
      plt.hist(
        dist, 
        bins=bins, 
        alpha=0.6,
        label=f"{self.labels[i]}", 
        # density=True,  # to match KDE style in scale
        edgecolor='black'
      )

    plt.title("Comparison of Final Points Distributions", fontsize=12)
    plt.xlabel("Total Final Points", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{self.DIRECTORY}/comparison_distribution.png")
    plt.show()

  def _plot_comparison_weekly(self, cumulative_points_list, cumulative_best_list):
    gw_range = list(range(1, len(cumulative_points_list[0]) + 1))
    plt.figure(figsize=(12, 6))
    for i, cumulative_points in enumerate(cumulative_points_list):
      plt.plot(gw_range, cumulative_points, label=f"{self.labels[i]} Avg", marker='o')
    for i, best_points in enumerate(cumulative_best_list):
      plt.plot(gw_range, best_points, label=f"{self.labels[i]} Best", linestyle='--', marker='x')
    plt.title("Cumulative Points Comparison (Average vs Best)", fontsize=12)
    plt.xlabel("Gameweek", fontsize=10)
    plt.ylabel("Cumulative Points", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{self.DIRECTORY}/cumulative_comparison.png")
    plt.show()

  def _plot_comparison_weekly_points(self, avg_points_list, best_points_list):
    gw_range = list(range(1, len(avg_points_list[0]) + 1))
    plt.figure(figsize=(12, 6))
    for i, avg_points in enumerate(avg_points_list):
      plt.plot(gw_range, avg_points, label=f"{self.labels[i]} Avg", marker='o')
    plt.title("Weekly Points Comparison", fontsize=12)
    plt.xlabel("Gameweek", fontsize=10)
    plt.ylabel("Points", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{self.DIRECTORY}/weekly_comparison.png")
    plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run the model with optional chip strategies.")

  parser.add_argument("--season", type=str, default="2023-24")
  transfers_strategies=['simple', 'weighted']
  parser.add_argument(
    "--transfers", type=str, choices=transfers_strategies, default='simple',
    help="Strategy to calculate the fitness of transfer candidates. Options: 'simple', 'weighted'"
  )
  parser.add_argument("--iterations", type=int, default=100)
  parser.add_argument("--load_saved", action="store_true")
  args = parser.parse_args()

  simulation_evaluation = SimulationEvaluation(
    season=args.season,
    iterations=args.iterations,
    transfers_strategy=args.transfers,
    load_saved=args.load_saved
  )
  simulation_evaluation.evaluate()