import matplotlib.pyplot as plt
import os

class TransfersEvaluation:
  def __init__(self, gw_data, transfers_strategy):
    self.gw_data = gw_data
    self.transfers_strategy = transfers_strategy

    directory_path = f'plots/transfers/tests/{self.transfers_strategy}'
    os.makedirs(directory_path, exist_ok=True)
    self.DIRECTORY = directory_path

  def evaluate(self, histories, best_history):
    grouped_histories = {}
    for history in histories:
      for gw in history.keys():
        grouped_histories.setdefault(gw, [])
        for transfer in history[gw]:
          grouped_histories[gw].append(transfer)

    evaluation_results = []
    best_point_changes = []
    avg_xP_changes = []
    avg_point_changes = []

    total_xP_change = 0
    total_point_change = 0
    beneficial_transfers = 0
    detrimental_transfers = 0

    for gw in sorted(grouped_histories.keys()):
      gw_data = self.gw_data[self.gw_data['GW'] == gw]

      gw_xP_deltas = []
      gw_point_deltas = []

      for transfer in grouped_histories[gw]:
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
      avg_xP_changes.append(avg_xP)
      avg_point_changes.append(avg_points)

      # Calculate best possible point change
      if gw in best_history:
        best_point_change = 0
        for transfer in best_history[gw]:
          player_in = transfer['in']
          player_out = transfer['out']
          best_point_change += player_in['total_points'] - player_out['total_points']
      else:
        best_point_change = 0

      best_point_changes.append(best_point_change)

      evaluation_results.append({
        'gameweek': gw,
        'average_xP_change': avg_xP,
        'average_point_change': avg_points,
        'transfers_count': len(grouped_histories[gw]),
      })

      total_xP_change += avg_xP
      total_point_change += avg_points

    avg_xP_change = total_xP_change / len(grouped_histories) if grouped_histories else 0
    avg_point_change = total_point_change / len(grouped_histories) if grouped_histories else 0
    total_best_points = sum(best_point_changes)
    average_best = total_best_points / len(best_point_changes) if best_point_changes else 0

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

    gameweeks = [res['gameweek'] for res in evaluation_results]

    self._plot_transfer_impact(gameweeks, avg_xP_changes, avg_point_changes, best_point_changes)
    self._plot_cumulative_gains(gameweeks, avg_point_changes, best_point_changes)
    self._plot_transfer_outcomes(beneficial_transfers, detrimental_transfers)
    self._plot_point_change_distribution(avg_point_changes)

    return {
      'weekly_evaluation': evaluation_results,
      'summary': evaluation_summary
    }

  def _plot_transfer_impact(self, gameweeks, avg_xP_changes, avg_point_changes, best_point_changes):
    plt.figure(figsize=(12, 6))
    plt.plot(gameweeks, avg_xP_changes, marker='o', label='Average xP Change')
    plt.plot(gameweeks, avg_point_changes, marker='x', label='Average Actual Point Change')
    plt.plot(gameweeks, best_point_changes, marker='s', linestyle='--', label='Best Result Point Change')
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    plt.title('Average Transfer Impact Per Gameweek')
    plt.xlabel('Gameweek')
    plt.ylabel('Points Gained')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{self.DIRECTORY}/transfer_impact_over_gameweeks.png')
    plt.show()

  def _plot_cumulative_gains(self, gameweeks, avg_point_changes, best_point_changes):
    cumulative_mean = []
    cumulative_best = []
    cum_mean, cum_best = 0, 0

    for i in range(len(gameweeks)):
      cum_mean += avg_point_changes[i]
      cum_best += best_point_changes[i]
      cumulative_mean.append(cum_mean)
      cumulative_best.append(cum_best)

    plt.figure(figsize=(12, 6))
    plt.plot(gameweeks, cumulative_mean, label='Cumulative Mean Points', marker='o')
    plt.plot(gameweeks, cumulative_best, label='Cumulative Best Points', marker='s', linestyle='--')
    plt.title('Cumulative Gains: Mean vs Best Transfers')
    plt.xlabel('Gameweek')
    plt.ylabel('Cumulative Points')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{self.DIRECTORY}/cumulative_mean_vs_best_gains.png')
    plt.show()

  def _plot_transfer_outcomes(self, beneficial_transfers, detrimental_transfers):
    plt.figure(figsize=(6, 5))
    plt.bar(['Beneficial', 'Detrimental'], [beneficial_transfers, detrimental_transfers], color=['green', 'red'])
    plt.title('Transfer Outcomes')
    plt.ylabel('Number of Transfers')
    plt.tight_layout()
    plt.savefig(f'{self.DIRECTORY}/transfer_outcomes.png')
    plt.show()

  def _plot_point_change_distribution(self, avg_point_changes):
    plt.figure(figsize=(10, 5))
    plt.hist(avg_point_changes, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Average Point Changes from Transfers')
    plt.xlabel('Point Change')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{self.DIRECTORY}/point_change_distribution.png')
    plt.show()