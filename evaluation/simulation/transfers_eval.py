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
from pathlib import Path
import pickle

class TransfersEvaluation:
  def __init__(self, season, target='xP', iterations=50, load_saved=False):
    self.season = season
    self.target = target
    self.ITERATIONS = iterations
    self.load_saved = load_saved

    self.data_loader = DataLoader()
    self.gw_data = self.data_loader.get_gw_predictions(season)

    self.strategies = ['simple', 'weighted']
    self.DIRECTORY = f'plots/transfers/test'
    os.makedirs(self.DIRECTORY, exist_ok=True)

  def _run_simulation(self, strategy):
    simulation = Simulation(
      season=self.season,
      transfers_strategy=strategy,
      debug=False
    )
    total_points, sim_histories = simulation.simulate_season()
    return total_points, sim_histories

  def evaluate(self):
    all_results = {}
    result_dir = Path(f"data/cached/transfers/{self.season}")
    result_dir.mkdir(parents=True, exist_ok=True)

    if self.load_saved:
      print("Loading saved evaluation results...")
      for strategy in self.strategies:
        with open(result_dir / f"{strategy}_evaluation.json", "r") as f:
          all_results[strategy] = json.load(f)
    else:
      for strategy in self.strategies:
        print(f"\nEvaluating strategy: {strategy.upper()}\n")
        transfer_histories = []
        teams_histories = []
        points_histories = []
        
        best_sim_points = 0
        best_sim_histories = None

        for i in range(self.ITERATIONS):
          print(f"Running simulation {i + 1}")
          sim_points, sim_histories = self._run_simulation(strategy)

          sim_transfers = sim_histories['transfers']
          sim_teams = sim_histories['teams']
          sim_points_per_gw = sim_histories['points']

          transfer_histories.append(sim_transfers)
          teams_histories.append(sim_teams)
          points_histories.append(sim_points_per_gw)

          if best_sim_points < sim_points:
            best_sim_points = sim_points
            best_sim_histories = sim_transfers
            best_sim_gw_points = sim_points_per_gw

        evaluation = self._evaluate_transfers(
          transfer_histories,
          best_sim_histories,
          strategy,
          points_histories,
          best_sim_gw_points
        )

        all_results[strategy] = evaluation

        with open(result_dir / f"{strategy}_evaluation.json", "w") as f:
          json.dump(evaluation, f, indent=2, default=str)

        with open(result_dir / f"{strategy}_evaluation_metrics.pkl", "wb") as f:
          evaluation_metrics = {
            'transfer_histories': transfer_histories,
            'best_sim_histories': best_sim_histories,
            'strategy': strategy,
            'points_histories': points_histories
          }
          pickle.dump(evaluation_metrics, f)

    self._plot_final_point_distributions(all_results)
    self._plot_points_boxplot(all_results)
    
    self._print_summary_table(all_results)
    return all_results

  def _print_summary_table(self, all_results):
    print("\nTransfers - Summary Comparison")
    print(f"{'Strategy':<15} {'Avg Points':>12} {'Std Dev':>10} {'Best Points':>12}")
    for strategy, result in all_results.items():
      final_points = [float(p) for p in result['final_points_distribution']]
      avg = sum(final_points) / len(final_points)
      std = pd.Series(final_points).std()
      best = max(final_points)
      label = 'xP' if strategy == 'simple' else 'Weighted'
      print(f"{label:<15} {avg:>12.2f} {std:>10.2f} {best:>12.0f}")

  def _evaluate_transfers(self, histories, best_history, strategy, points_histories, best_gw_points):
    grouped_histories = {}
    for history in histories:
      for gw in history.keys():
        grouped_histories.setdefault(gw, [])
        for transfer in history[gw]:
          grouped_histories[gw].append(transfer)

    evaluation_results = []
    best_point_changes = [0] * 38
    avg_xP_changes = [0] * 38
    avg_point_changes = [0] * 38

    total_xP_change = 0
    total_point_change = 0
    beneficial_transfers = 0
    detrimental_transfers = 0

    for gw in range(1, 39):  # Fill 0s for gameweeks with no transfers
      gw_data = self.gw_data[self.gw_data['GW'] == gw]

      gw_xP_deltas = []
      gw_point_deltas = []

      for transfer in grouped_histories.get(gw, []):
        player_in = transfer['in']
        player_out = transfer['out']

        xP_change = player_in['xP'] - player_out['xP']
        point_change = player_in['total_points'] - player_out['total_points']

        gw_xP_deltas.append(xP_change)
        gw_point_deltas.append(point_change)

        if point_change > 0:
          beneficial_transfers += 1
        elif point_change < 0:
          detrimental_transfers += 1

      avg_xP = sum(gw_xP_deltas) / len(gw_xP_deltas) if gw_xP_deltas else 0
      avg_points = sum(gw_point_deltas) / len(gw_point_deltas) if gw_point_deltas else 0
      avg_xP_changes[gw - 1] = avg_xP
      avg_point_changes[gw - 1] = avg_points

      if gw in best_history and best_history[gw]:
        best_point_change = 0
        for transfer in best_history[gw]:
          player_in = transfer['in']
          player_out = transfer['out']
          best_point_change += player_in['total_points'] - player_out['total_points']
      else:
        best_point_change = 0

      best_point_changes[gw - 1] = best_point_change

      evaluation_results.append({
        'gameweek': gw,
        'average_xP_change': avg_xP,
        'average_point_change': avg_points,
        'transfers_count': len(grouped_histories.get(gw, [])),
      })

      total_xP_change += avg_xP
      total_point_change += avg_points

    avg_xP_change = total_xP_change / len(grouped_histories) if grouped_histories else 0
    avg_point_change = total_point_change / len(grouped_histories) if grouped_histories else 0
    total_best_points = sum(best_point_changes)
    average_best = total_best_points / len(best_point_changes) if best_point_changes else 0

    gw_order = sorted(points_histories[0].keys())
    average_points_per_gw = []
    cumulative_avg_points = []
    std_dev_per_gw = []
    cum_total = 0

    for gw in gw_order:
      gw_values = [history.get(gw, 0) for history in points_histories]
      gw_avg = sum(gw_values) / len(gw_values)
      gw_std = pd.Series(gw_values).std()
      average_points_per_gw.append(gw_avg)
      std_dev_per_gw.append(gw_std)
      cum_total += gw_avg
      cumulative_avg_points.append(cum_total)

    final_totals = [sum(history.values()) for history in points_histories]
    
    # Best simulation points per GW and cumulative
    best_points_per_gw = []
    best_cumulative_points = []
    cum_best_total = 0

    for gw in range(1, 39):
      gw_points = best_gw_points.get(gw, 0)
      best_points_per_gw.append(gw_points)
      cum_best_total += gw_points
      best_cumulative_points.append(cum_best_total)

    evaluation_summary = {
      'total_gameweeks': len(grouped_histories),
      'total_transfers': beneficial_transfers + detrimental_transfers,
      'beneficial_transfers': beneficial_transfers,
      'detrimental_transfers': detrimental_transfers,
      'average_xP_change_per_gw': avg_xP_change,
      'average_point_change_per_gw': avg_point_change,
      'total_best_points': total_best_points,
      'average_best_points_per_gw': average_best
    }

    print(evaluation_summary)

    return {
      'weekly_evaluation': evaluation_results,
      'summary': evaluation_summary,
      'best_point_changes': best_point_changes,
      'avg_point_changes': avg_point_changes,
      'points_per_gw': average_points_per_gw,
      'cumulative_points': cumulative_avg_points,
      'best_cumulative_points': best_cumulative_points,
      'stddev_per_gw': std_dev_per_gw,
      'final_points_distribution': final_totals,
      'raw_points': points_histories
    }

  def _plot_final_point_distributions(self, all_results):
    plt.figure(figsize=(10, 6))
    for strategy, result in all_results.items():
      plt.hist(
        result['final_points_distribution'],
        bins=15,
        alpha=0.6,
        label=f"{'xP' if strategy == 'simple' else 'Weighted'}",
        edgecolor='black'
      )
    plt.title('Distribution of Final Total Points by Player Selection Strategy')
    plt.xlabel('Total Final Points')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{self.DIRECTORY}/final_points_distribution.png")
    plt.show()

  def _plot_points_boxplot(self, all_results):
    plt.figure(figsize=(8, 6))
    data = []
    for strategy, result in all_results.items():
      for total in result['final_points_distribution']:
        data.append({'Strategy': strategy, 'Final Points': total})
    df = pd.DataFrame(data)
    df['Strategy'] = df['Strategy'].replace({'simple': 'xP', 'weighted': 'Weighted'})
    sns.boxplot(x='Strategy', y='Final Points', data=df)
    # Boxplot
    plt.title('Final Points Distribution by Player Selection Strategy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{self.DIRECTORY}/boxplot_final_points.png")
    plt.show()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--season", type=str, default="2023-24")
  parser.add_argument("--target", type=str, default='xP', choices=['fpl_xP', 'xP'])
  parser.add_argument("--iterations", type=int, default=50)
  parser.add_argument("--load_saved", action="store_true", help="Load saved evaluation results instead of simulating")

  args = parser.parse_args()

  evaluator = TransfersEvaluation(
    season=args.season,
    target=args.target,
    iterations=args.iterations,
    load_saved=args.load_saved
  )
  evaluator.evaluate()
