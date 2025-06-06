import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data.data_loader import DataLoader
from utils.feature_selector import FeatureSelector
import argparse
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from utils.model_types import ModelType
from pathlib import Path
from tabulate import tabulate

class ModelEvaluation:
  def __init__(
    self, 
    season, 
    time_steps, 
    model_type, 
    include_prev_season=False, 
    include_fbref=False, 
    include_season_aggs=False, 
    include_teams=False,
    top_features=None,
    gw_lamda_decay=0.02
  ):
    self.season = season
    self.time_steps = time_steps
    self.model_type = model_type
    self.feature_selector = FeatureSelector()

    self.include_prev_season = include_prev_season
    self.include_fbref = include_fbref
    self.include_season_aggs = include_season_aggs
    self.include_teams = include_teams
    self.gw_lamda_decay = gw_lamda_decay
    self.top_features = top_features

    self.evaluation_results = []  # Stores results for CSV export

    gw_decay_str = str(self.gw_lamda_decay).replace('.', '_')
    self.FILE_NAME = f"steps_{self.time_steps}_prev_season_{self.include_prev_season}_fbref_{self.include_fbref}_season_aggs_{self.include_season_aggs}_teams_{self.include_teams}_gw_decay_{gw_decay_str}_top_features_{self.top_features}"

  def evaluate(self):
    self.feature_selector = FeatureSelector()
    evaluation_df = self._load_predictions()
    evaluation_df = evaluation_df.dropna(subset=[self.feature_selector.EXPECTED])
    
    # General evaluation
    ae, mae, mse, rmse = self.score_model(evaluation_df, self.feature_selector.EXPECTED)
    baseline_ae, baseline_mae, baseline_mse, baseline_rmse = self.score_model(evaluation_df, self.feature_selector.BASELINE)

    print('--- Expected ---')
    print(f'Absolute Error: {ae}')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')

    print('\n--- Baseline ---')
    print(f'Absolute Error: {baseline_ae}')
    print(f'Mean Absolute Error: {baseline_mae}')
    print(f'Mean Squared Error: {baseline_mse}')
    print(f'Root Mean Squared Error: {baseline_rmse}')

    self.evaluation_results.append(["Overall", "General", round(ae, 2), round(mse, 2), round(baseline_ae, 2), round(baseline_mse, 2)])
    
    # self.evaluate_grouclped_by_points(evaluation_df)
    self.evaluate_grouped_by_position_costs(evaluation_df)

  def evaluate_grouped_by_position_costs(self, evaluation_df):
    print("\n--- Evaluation Grouped by Cost Bands (Threshold-Based) ---")

    cost_thresholds = {
      'GK': {'budget_max': 45, 'premium_min': 50},
      'DEF': {'budget_max': 45, 'premium_min': 55},
      'MID': {'budget_max': 60, 'premium_min': 85},
      'FWD': {'budget_max': 60, 'premium_min': 90}
    }

    positions = ['GK', 'DEF', 'MID', 'FWD']
    all_expected_errors = []
    all_baseline_errors = []

    for pos in positions:
      pos_df = evaluation_df[evaluation_df['position'] == pos].copy()

      if pos_df.empty:
        continue

      # Ensure cost is numeric
      pos_df[self.feature_selector.COST] = pos_df[self.feature_selector.COST].astype(float)
      thresholds = cost_thresholds[pos]

      # Assign cost band using apply
      def get_cost_band(row):
        cost = row[self.feature_selector.COST]
        if cost <= thresholds['budget_max']:
          return 'Budget'
        elif cost >= thresholds['premium_min']:
          return 'Premium'
        else:
          return 'Mid'

      pos_df['cost_band'] = pos_df.apply(get_cost_band, axis=1)

      for band in ['Budget', 'Mid', 'Premium']:
        group_df = pos_df[pos_df['cost_band'] == band]
        if group_df.empty:
          continue

        expected_errors = self.score_model(group_df, self.feature_selector.EXPECTED)
        baseline_errors = self.score_model(group_df, self.feature_selector.BASELINE)

        all_expected_errors.append({
          'Position': pos,
          'Cost Band': band,
          'AE': expected_errors[0],
          'MAE': expected_errors[1],
          'MSE': expected_errors[2],
          'RMSE': expected_errors[3]
        })

        all_baseline_errors.append({
          'Position': pos,
          'Cost Band': band,
          'AE': baseline_errors[0],
          'MAE': baseline_errors[1],
          'MSE': baseline_errors[2],
          'RMSE': baseline_errors[3]
        })

    # Convert to DataFrames and display as joint tables
    expected_df = pd.DataFrame(all_expected_errors)
    baseline_df = pd.DataFrame(all_baseline_errors)

    print("\n--- Expected Errors by Cost Band ---")
    self.log_grouped_errors(expected_df, 'position-costband-expected')

    print("\n--- Baseline Errors by Cost Band ---")
    self.log_grouped_errors(baseline_df, 'position-costband-baseline')

  def score_model(self, evaluation_df, expected):
    targets = evaluation_df[self.feature_selector.TARGET]
    expected = evaluation_df[expected]
    
    ae = np.abs(targets - expected).mean()
    mae = mean_absolute_error(targets, expected)
    mse = mean_squared_error(targets, expected)
    rmse = np.sqrt(mse)
    
    return np.round(ae, 2), np.round(mae, 2), np.round(mse, 2), np.round(rmse, 2)

  def log_grouped_errors(self, grouped_errors, label):
    print(tabulate(grouped_errors, headers='keys', tablefmt='pretty'))

  def _load_predictions(self):
    project_root = Path(__file__).resolve().parent.parent
    predictions_dir = f"{self.model_type.value}/{self.FILE_NAME}"
    directory = project_root / 'predictions' / predictions_dir / 'gws' / self.season

    if directory.exists():
      path = Path(directory)
      predictions_df = pd.concat([pd.read_csv(csv_file) for csv_file in path.glob("*.csv")], ignore_index=True)
      return predictions_df
    else:
      raise Exception(f"Predictions not found: {self.FILE_NAME}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run the model evaluation.')
  parser.add_argument('--season', type=str, default='2023-24')
  parser.add_argument('--steps', type=int, default=5)
  parser.add_argument('--prev_season', action='store_true')
  parser.add_argument('--model', type=str, choices=[m.value for m in ModelType])
  parser.add_argument('--fbref', action='store_true')
  parser.add_argument('--season_aggs', action='store_true')
  parser.add_argument('--gw_decay', type=float, default=0.02, help='The lambda decay applied in gw decay')
  parser.add_argument('--teams', action='store_true')
  parser.add_argument('--top_features', type=int, nargs='?', help='Time step for data window. Defaults to 7 if not provided or null.')

  args = parser.parse_args()

  model_evaluation = ModelEvaluation(
    season=args.season, 
    time_steps=args.steps, 
    model_type=ModelType(args.model), 
    include_prev_season=args.prev_season, 
    include_fbref=args.fbref, 
    include_season_aggs=args.season_aggs, 
    gw_lamda_decay=args.gw_decay,
    top_features=args.top_features,
    include_teams=args.teams
  )
  model_evaluation.evaluate()
