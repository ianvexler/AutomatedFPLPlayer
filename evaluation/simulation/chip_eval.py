import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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

class ChipEvaluation:
  def __init__(
    self, 
    season, 
    chip, 
    target='xP', 
    iterations=50, 
    load_saved=False
  ):
    self.season = season
    self.chip = chip
    self.target = target
    self.ITERATIONS = iterations
    self.load_saved = load_saved

    self.data_loader = DataLoader()
    self.gw_data = self.data_loader.get_gw_predictions(season)

    self.DIRECTORY = f'plots/{chip.value}/test'
    os.makedirs(self.DIRECTORY, exist_ok=True)

  def _run_simulation(self, strategy):
    all_strategies = {
      Chip.TRIPLE_CAPTAIN: self._get_chip_strategy(Chip.TRIPLE_CAPTAIN),
      Chip.WILDCARD: self._get_chip_strategy(Chip.WILDCARD),
      Chip.FREE_HIT: self._get_chip_strategy(Chip.FREE_HIT),
      Chip.BENCH_BOOST: self._get_chip_strategy(Chip.BENCH_BOOST)
    }

    chip_strategy = {
      chip: random.choice(strategies)
      for chip, strategies in all_strategies.items()
    }
    chip_strategy[self.chip] = strategy

    simulation = Simulation(
      season=self.season,
      transfers_strategy='weighted',
      chip_strategy=chip_strategy,
      debug=False
    )
    total_points, sim_histories = simulation.simulate_season()
    return total_points, sim_histories, chip_strategy

  def _get_chip_strategy(self, chip):
    match chip.value:
      case Chip.TRIPLE_CAPTAIN.value:
        return ["conservative"]
      case Chip.WILDCARD.value:
        return ["asap", "wait", "double_gw"]
      case Chip.FREE_HIT.value:
        return ["double_gw", "blank_gw"]
      case Chip.BENCH_BOOST.value:
        return ["double_gw", "with_wildcard"]
    raise Exception('Invalid Chip')

  def evaluate(self):
    random.seed(42)
    all_results = {}
    result_dir = Path(f"data/cached/chips/{self.chip}/{self.season}")
    result_dir.mkdir(parents=True, exist_ok=True)
    strategies = self._get_chip_strategy(self.chip)

    if self.load_saved:
      for strategy in strategies:
        with open(result_dir / f"{strategy}_evaluation_metrics.pkl", "rb") as f:
          evaluation_metrics = pickle.load(f)
          results = self._evaluate_chip_strategy(evaluation_metrics)
          all_results[strategy] = results
    else:
      for strategy in strategies:
        all_sim_points = []
        all_sim_histories = []
        all_chip_strategies = []
        best_sim_points = 0
        best_sim_histories = None
        best_sim_chip_strategy = None

        for i in range(self.ITERATIONS):
          sim_points, sim_histories, chip_strategy = self._run_simulation(strategy)
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
          'strategy': strategy,
          'all_sim_points': all_sim_points,
          'all_sim_histories': all_sim_histories,
          'all_chip_strategies': all_chip_strategies
        }

        with open(result_dir / f"{strategy}_evaluation_metrics.pkl", "wb") as f:
          pickle.dump(evaluation_metrics, f)

        results = self._evaluate_chip_strategy(evaluation_metrics)
        all_results[strategy] = results

    self._plot_combined_chip_strategies(all_results)
    self._plot_final_point_distributions(all_results)
    self._plot_final_points_boxplot(all_results)
    self._print_summary_table(all_results)
    # self._generate_all_alternative_plots(all_results)

  def _evaluate_chip_strategy(self, evaluation_metrics):
    histories = evaluation_metrics["all_sim_histories"]
    best_points = evaluation_metrics["best_sim_points"]
    strategy = evaluation_metrics["strategy"]

    chip_usages_per_gw = [0] * 38
    chip_effect_points = [0] * 38
    chip_effect_counts = [0] * 38
    points_histories = [h.get("points", {}) for h in histories]
    gw_order = sorted(points_histories[0].keys())
    average_points_per_gw = []
    std_dev_per_gw = []
    cumulative_avg_points = []
    cum_total = 0

    for gw in gw_order:
      gw_values = [history.get(gw, 0) for history in points_histories]
      avg = sum(gw_values) / len(gw_values)
      std = pd.Series(gw_values).std()
      average_points_per_gw.append(avg)
      std_dev_per_gw.append(std)
      cum_total += avg
      cumulative_avg_points.append(cum_total)

    for sim in histories:
      chips = sim.get("chips", [])
      points = sim.get("points", {})
      for chip_entry in chips:
        if chip_entry == self.chip.value:
          gw = chips[chip_entry]
          used_points = points.get(gw, 0)
          if 1 <= gw <= 38:
            chip_usages_per_gw[gw - 1] += 1
            chip_effect_points[gw - 1] += used_points
            chip_effect_counts[gw - 1] += 1

    avg_chip_effect = [
      chip_effect_points[i] / chip_effect_counts[i] if chip_effect_counts[i] > 0 else 0
      for i in range(38)
    ]

    evaluation_results = []
    for gw in range(1, 39):
      evaluation_results.append({
        'gameweek': gw,
        'chip_usages': chip_usages_per_gw[gw - 1],
        'average_points_with_chip': avg_chip_effect[gw - 1]
      })

    sim_points = [float(x) for x in evaluation_metrics['all_sim_points']]
    final_totals = [sum(sim.values()) for sim in points_histories]

    summary = {
      'strategy': strategy,
      'total_chip_usages': sum(chip_usages_per_gw),
      'average_points_when_chip_used': (
        sum(chip_effect_points) / sum(chip_effect_counts)
        if sum(chip_effect_counts) > 0 else 0
      ),
      'best_sim_points': best_points,
      'avg_sim_points': sum(sim_points) / len(sim_points) if sim_points else 0,
      'stddev_sim_points': statistics.stdev(sim_points) if len(sim_points) > 1 else 0
    }

    return {
      'weekly_evaluation': evaluation_results,
      'summary': summary,
      'points_per_gw': average_points_per_gw,
      'cumulative_points': cumulative_avg_points,
      'stddev_per_gw': std_dev_per_gw,
      'raw_points': points_histories,
      'final_points_distribution': final_totals
    }

  def _plot_combined_chip_strategies(self, all_results):
    gameweeks = list(range(1, 39))
    plt.figure(figsize=(12, 6))
    for strategy, results in all_results.items():
      plt.plot(gameweeks, results['points_per_gw'], label=strategy.replace('_', ' ').replace('gw', 'GW').capitalize().replace('_', ' ').replace('gw', 'GW').capitalize() + " avg", marker='o')
    plt.title(f'{self.humanize(self.chip.value)} - Average Points Per Gameweek by Strategy')
    plt.xlabel('Gameweek')
    plt.ylabel('Points')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(self.DIRECTORY, 'combined_chip_strategies_1.png'))
    # plt.show()

    plt.figure(figsize=(12, 6))
    for strategy, results in all_results.items():
      plt.plot(gameweeks, results['cumulative_points'], label=strategy.replace('_', ' ').replace('gw', 'GW').capitalize().replace('_', ' ').replace('gw', 'GW').capitalize() + " cumulative", marker='o')
    plt.title(f'{self.humanize(self.chip.value)} - Cumulative Points Over Gameweeks')
    plt.xlabel('Gameweek')
    plt.ylabel('Cumulative Points')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(self.DIRECTORY, 'combined_chip_strategies_2.png'))
    # plt.show()

  def _plot_final_point_distributions(self, all_results):
    plt.figure(figsize=(10, 6))
    for strategy, result in all_results.items():
      plt.hist(
        result['final_points_distribution'],
        bins=15,
        alpha=0.6,
        label=strategy.replace('_', ' ').replace('gw', 'GW').capitalize(),
        edgecolor='black'
      )
    plt.title(f'{self.chip.value} - Final Total Points Distribution by Strategy')
    plt.xlabel('Total Points')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(self.DIRECTORY, 'final_point_distributions.png'))
    # plt.show()

  def _plot_final_points_boxplot(self, all_results):
    data = []
    for strategy, result in all_results.items():
      for score in result['final_points_distribution']:
        data.append({'Strategy': strategy, 'Final Points': score})
    df = pd.DataFrame(data)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Strategy', y='Final Points', data=df)
    plt.title(f'{self.humanize(self.chip.value)} - Final Points Distribution by Strategy', fontsize=12)
    plt.xlabel('Strategy', fontsize=10)
    plt.ylabel('Final Points', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(self.DIRECTORY, 'final_points_boxplot.png'))
    # plt.show()

  def _print_summary_table(self, all_results):
    print(f"\n{self.humanize(self.chip.value)} - Summary Comparison")
    print(f"{'Strategy':<15} {'Avg Points':>12} {'Std Dev':>10} {'Best Points':>12}")
    for strategy, result in all_results.items():
      summary = result['summary']
      print(f"{strategy:<15} {summary['avg_sim_points']:>12.2f} {summary['stddev_sim_points']:>10.2f} {summary['best_sim_points']:>12.0f}")

  def humanize(self, s):
    return s.replace('_', ' ').title()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--season", type=str, default="2023-24")
  parser.add_argument("--target", type=str, default='xP', choices=['fpl_xP', 'xP'])
  parser.add_argument("--iterations", type=int, default=50)
  parser.add_argument("--load_saved", action="store_true")
  args = parser.parse_args()
  
  for chip in [Chip.WILDCARD, Chip.FREE_HIT, Chip.BENCH_BOOST]:
    evaluator = ChipEvaluation(
      season=args.season,
      target=args.target,
      iterations=args.iterations,
      load_saved=args.load_saved,
      chip=chip
    )
    evaluator.evaluate()
