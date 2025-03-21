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

class Evaluate:
  def __init__(self, season, time_steps, model_type, include_prev_season=False, include_fbref=False, include_season_aggs=False):
    self.season = season
    self.time_steps = time_steps
    self.model_type = model_type
    self.feature_selector = FeatureSelector()
    self.include_prev_season = include_prev_season
    self.include_fbref = include_fbref
    self.include_season_aggs = include_season_aggs

    self.evaluation_results = []  # Stores results for CSV export

    self.FILE_NAME = f"steps_{self.time_steps}_prev_season_{self.include_prev_season}_fbref_{self.include_fbref}_season_aggs_{self.include_season_aggs}"

  def evaluate_model(self):
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
    self.evaluate_grouped_by_costs(evaluation_df)
    self.export_evaluation_to_csv()

  def evaluate_grouped_by_points(self, evaluation_df):
    print("\n--- Evaluation Grouped by Points ---")
    points_q25 = evaluation_df[self.feature_selector.TARGET].quantile(0.25)
    points_q75 = evaluation_df[self.feature_selector.TARGET].quantile(0.75)

    evaluation_df['points_group'] = np.select(
      [evaluation_df[self.feature_selector.TARGET] <= points_q25,
       evaluation_df[self.feature_selector.TARGET] >= points_q75],
      ['Bottom 25%', 'Top 25%'],
      default='Middle 50%'
    )

    grouped_errors = evaluation_df.groupby('points_group').apply(
      lambda df: pd.Series(self.score_model(df, self.feature_selector.EXPECTED),
                           index=['AE', 'MAE', 'MSE', 'RMSE']), 
      include_groups=False
    )

    print('\nExpected')
    self.log_grouped_errors(grouped_errors, 'points')

    baseline_errors = evaluation_df.groupby('points_group').apply(
      lambda df: pd.Series(self.score_model(df, self.feature_selector.BASELINE),
                           index=['AE', 'MAE', 'MSE', 'RMSE']),
      include_groups=False
    )

    print('\nBaseline')
    self.log_grouped_errors(baseline_errors, 'points')

  def evaluate_grouped_by_costs(self, evaluation_df):
    print("\n--- Evaluation Grouped by Costs ---")
    cost_q25 = evaluation_df[self.feature_selector.COST].quantile(0.25)
    cost_q75 = evaluation_df[self.feature_selector.COST].quantile(0.75)

    evaluation_df['cost_group'] = np.select(
      [evaluation_df[self.feature_selector.COST] <= cost_q25,
       evaluation_df[self.feature_selector.COST] >= cost_q75],
      ['Bottom 25%', 'Top 25%'],
      default='Middle 50%'
    )

    grouped_errors = evaluation_df.groupby('cost_group').apply(
      lambda df: pd.Series(self.score_model(df, self.feature_selector.EXPECTED),
                           index=['AE', 'MAE', 'MSE', 'RMSE']),
      include_groups=False
    )

    print('\nExpected')
    self.log_grouped_errors(grouped_errors, 'cost')

    baseline_errors = evaluation_df.groupby('cost_group', group_keys=False).apply(
      lambda df: pd.Series(self.score_model(df, self.feature_selector.BASELINE),
                           index=['AE', 'MAE', 'MSE', 'RMSE']),
      include_groups=False
    )

    print('\nBaseline')
    self.log_grouped_errors(baseline_errors, 'cost')

  def export_evaluation_to_csv(self):
    evaluations_dir = "evaluations"
    os.makedirs(evaluations_dir, exist_ok=True)
    csv_filename = f"evaluation_{self.season}_{self.FILE_NAME}.csv"
    csv_path = os.path.join(evaluations_dir, csv_filename)

    df = pd.DataFrame(self.evaluation_results, columns=["Group", "Metric", "Model AE", "Model MSE", "Baseline AE", "Baseline MSE"])
    df.to_csv(csv_path, index=False)
    print(f"Evaluation results saved to {csv_path}")

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
    predictions_dir = f"{self.model_type.value}/{self.FILE_NAME}"
    directory = f'predictions/{predictions_dir}/gws/{self.season}'
    
    if os.path.exists(directory):
      path = Path(directory)
      predictions_df = pd.concat([pd.read_csv(csv_file) for csv_file in path.glob("*.csv")], ignore_index=True)
      return predictions_df
    else:
      raise Exception(f"Predictions not found")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run the model evaluation.')
  parser.add_argument('--season', type=str, default='2024-25')
  parser.add_argument('--steps', type=int, default=5)
  parser.add_argument('--prev_season', action='store_true')
  parser.add_argument('--model', type=str, choices=[m.value for m in ModelType])
  parser.add_argument('--fbref', action='store_true')
  parser.add_argument('--season_aggs', action='store_true')
  args = parser.parse_args()
  evaluate = Evaluate(args.season, args.steps, ModelType(args.model), args.prev_season, args.fbref, args.season_aggs)
  evaluate.evaluate_model()
