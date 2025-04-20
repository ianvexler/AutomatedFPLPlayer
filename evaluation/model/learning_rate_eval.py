import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from sklearn.metrics import mean_squared_error
from utils.feature_selector import FeatureSelector
import argparse
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from utils.model_types import ModelType
from pathlib import Path
from tabulate import tabulate
import seaborn as sns
import joblib
from model import Model

class LearningRateEvaluation:
  def __init__(
    self, 
    season, 
    time_steps, 
    include_prev_season=False, 
    include_fbref=False, 
    include_season_aggs=False, 
    include_teams=False,
    gw_lamda_decay=0.02,
    n_features=20
  ):
    self.season = season
    self.time_steps = time_steps
    self.feature_selector = FeatureSelector()

    self.include_prev_season = include_prev_season
    self.include_fbref = include_fbref
    self.include_season_aggs = include_season_aggs
    self.include_teams = include_teams
    self.gw_lamda_decay = gw_lamda_decay
    self.n_features = n_features
    self.model = None

    self.FILE_NAME = f"steps_{self.time_steps}_prev_season_{self.include_prev_season}_fbref_{self.include_fbref}_season_aggs_{self.include_season_aggs}_teams_{self.include_teams}_gw_decay_{self.gw_lamda_decay}"

  def evaluate(self):
    learning_rates_to_test = [0.0001, 0.0003, 0.001, 0.003, 0.01]
    model_type = ModelType.LSTM
    results = []

    for lr in learning_rates_to_test:
      print(f"Training model with learning rate {lr}...")
      lstm_config = {
        'units': 32,
        'dropout': 0.3,
        'learning_rate': lr
      }

      self.model = Model(
        season=self.season,
        time_steps=self.time_steps,
        include_prev_season=self.include_prev_season,
        model_type=model_type,
        include_fbref=self.include_fbref,
        include_season_aggs=self.include_season_aggs,
        include_teams=self.include_teams,
        gw_lamda_decay=self.gw_lamda_decay,
        lstm_config=lstm_config
      )

      _, model_histories = self.model.train()

      for position in ['GK', 'DEF', 'MID', 'FWD']:
        history = model_histories[position]
        val_mae = min(history.history['val_mae'])
        val_mse = min(history.history['val_loss'])
        val_rmse = np.sqrt(val_mse)

        results.append({
          'position': position,
          'learning_rate': lr,
          'val_rmse': val_rmse,
          'val_mae': val_mae
        })

    df = pd.DataFrame(results)
    print("\nLearning Rate Evaluation Results:\n")
    print(tabulate(df, headers='keys', tablefmt='pretty', floatfmt=".4f"))

    plot_dir = f"plots/learning_rate/{self.FILE_NAME}"
    os.makedirs(plot_dir, exist_ok=True)

    for pos in ['GK', 'DEF', 'MID', 'FWD']:
      subset = df[df['position'] == pos]
      plt.figure(figsize=(8, 5))
      plt.plot(subset['learning_rate'], subset['val_rmse'], marker='o', label='RMSE')
      plt.plot(subset['learning_rate'], subset['val_mae'], marker='s', linestyle='--', label='MAE')
      plt.xlabel("Learning Rate")
      plt.ylabel("Error")
      plt.title(f"Validation Error vs Learning Rate - {pos}")
      plt.legend()
      plt.grid(True)
      plt.tight_layout()
      plt.savefig(f"{plot_dir}/learning_rate_performance_{pos.lower()}.png")
      plt.close()

    plt.figure(figsize=(10, 6))
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
      subset = df[df['position'] == pos]
      plt.plot(subset['learning_rate'], subset['val_rmse'], marker='o', label=f'{pos} RMSE')
      plt.plot(subset['learning_rate'], subset['val_mae'], marker='s', linestyle='--', label=f'{pos} MAE')
    plt.xlabel("Learning Rate")
    plt.ylabel("Error")
    plt.title("Validation Error vs Learning Rate (All Positions)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/learning_rate_performance_all_positions.png")
    plt.close()

    pivot_rmse = df.pivot(index='position', columns='learning_rate', values='val_rmse')
    pivot_mae = df.pivot(index='position', columns='learning_rate', values='val_mae')

    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot_rmse, annot=True, fmt=".4f", cmap="YlGnBu")
    plt.title("Validation RMSE (Learning Rate vs Position)")
    plt.xlabel("Learning Rate")
    plt.ylabel("Position")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/heatmap_rmse.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot_mae, annot=True, fmt=".4f", cmap="YlOrBr")
    plt.title("Validation MAE (Learning Rate vs Position)")
    plt.xlabel("Learning Rate")
    plt.ylabel("Position")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/heatmap_mae.png")
    plt.close()

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Run the model with optional training.')
  parser.add_argument('--steps', type=int, nargs='?', const=5, default=5, help='Time step for data window. Defaults to 7 if not provided or null.')
  parser.add_argument('--season', type=str, nargs='?', default='2023-24', help='Season to simulate in the format 20xx-yy.')
  parser.add_argument('--prev_season', action='store_true', help='Set this flag to include prev season data. Defaults to false.')
  parser.add_argument('--fbref', action='store_true', help='Include FBref data.')
  parser.add_argument('--season_aggs', action='store_true', help='Include season aggregate data.')
  parser.add_argument('--teams', action='store_true', help='Include teams data.')
  parser.add_argument('--gw_decay', type=float, default=0.02, help='The lambda decay applied in gw decay')
  parser.add_argument('--top_features', type=int, default=None)

  args = parser.parse_args()

  model_evaluation = LearningRateEvaluation(
    season=args.season,
    time_steps=args.steps,
    include_prev_season=args.prev_season,
    include_fbref=args.fbref,
    include_season_aggs=args.season_aggs,
    include_teams=args.teams,
    gw_lamda_decay=args.gw_decay,
    n_features=args.top_features)

  model_evaluation.evaluate()