import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data.data_loader import DataLoader
from utils.feature_selector import FeatureSelector
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from utils.model_types import ModelType
from pathlib import Path

class Evaluate:
  def __init__(self, season, steps, model_type, include_prev_season=False):
    self.season = season
    self.steps = steps
    self.model_type = model_type
    self.feature_selector = FeatureSelector()
    self.include_prev_season = include_prev_season
    self.evaluation_results = []  # Stores results for CSV export

  def evaluate_model(self):
    self.feature_selector = FeatureSelector()

    evaluation_df = self._load_predictions()
    evaluation_df = evaluation_df.dropna(subset=[self.feature_selector.EXPECTED])
    
    # General evaluation
    error, mse = self.score_model(evaluation_df, self.feature_selector.EXPECTED)
    baseline_error, baseline_mse = self.score_model(evaluation_df, self.feature_selector.BASELINE)

    print('--- Expected ---')
    print(f'Mean Absolute Error: {error}')
    print(f'Mean Squared Error: {mse}\n')

    print('--- Baseline ---')
    print(f'Mean Absolute Error: {baseline_error}')
    print(f'Mean Squared Error: {baseline_mse}\n')

    # Store general results
    self.evaluation_results.append(["Overall", "General", round(error, 2), round(mse, 2), round(baseline_error, 2), round(baseline_mse, 2)])

    # Run grouped evaluations
    self.evaluate_grouped_by_points(evaluation_df)
    self.evaluate_grouped_by_costs(evaluation_df)

    # Export results to CSV
    self.export_evaluation_to_csv()

  def evaluate_grouped_by_points(self, evaluation_df):
    print("\n--- Evaluation Grouped by Points ---")

    # Define quantiles for grouping
    points_q25 = evaluation_df[self.feature_selector.TARGET].quantile(0.25)
    points_q75 = evaluation_df[self.feature_selector.TARGET].quantile(0.75)

    # Assign group labels
    evaluation_df['points_group'] = np.select(
      [evaluation_df[self.feature_selector.TARGET] <= points_q25,
       evaluation_df[self.feature_selector.TARGET] >= points_q75],
      ['Bottom 25%', 'Top 25%'],
      default='Middle 50%'
    )

    # Compute model errors per group
    grouped_errors = evaluation_df.groupby('points_group')[[self.feature_selector.EXPECTED, self.feature_selector.TARGET]].apply(
      lambda df: self.score_model(df, self.feature_selector.EXPECTED)
    )

    # Compute baseline errors per group
    baseline_errors = evaluation_df.groupby('points_group')[[self.feature_selector.BASELINE, self.feature_selector.TARGET]].apply(
      lambda df: self.score_model(df, self.feature_selector.BASELINE)
    )

    self.log_grouped_errors(grouped_errors, 'points', baseline_errors)

  def evaluate_grouped_by_costs(self, evaluation_df):
    print("\n--- Evaluation Grouped by Costs ---")

    # Define quantiles for grouping
    cost_q25 = evaluation_df[self.feature_selector.COST].quantile(0.25)
    cost_q75 = evaluation_df[self.feature_selector.COST].quantile(0.75)

    # Assign group labels
    evaluation_df['cost_group'] = np.select(
      [evaluation_df[self.feature_selector.COST] <= cost_q25,
       evaluation_df[self.feature_selector.COST] >= cost_q75],
      ['Bottom 25%', 'Top 25%'],
      default='Middle 50%'
    )

    # Compute model errors per group
    grouped_errors = evaluation_df.groupby('cost_group')[[self.feature_selector.EXPECTED, self.feature_selector.TARGET]].apply(
      lambda df: self.score_model(df, self.feature_selector.EXPECTED)
    )

    # Compute baseline errors per group
    baseline_errors = evaluation_df.groupby('cost_group')[[self.feature_selector.BASELINE, self.feature_selector.TARGET]].apply(
      lambda df: self.score_model(df, self.feature_selector.BASELINE)
    )

    self.log_grouped_errors(grouped_errors, 'cost', baseline_errors)

  def score_model(self, evaluation_df, expected):
    # Extract targets and expected values
    targets = evaluation_df[self.feature_selector.TARGET]
    expected = evaluation_df[expected]
    
    error = mean_absolute_error(targets, expected)
    mse = mean_squared_error(targets, expected)

    return error, mse

  def log_grouped_errors(self, grouped_errors, label, baseline_errors):
    for group in grouped_errors.index:
      mae_model, mse_model = grouped_errors.loc[group]
      mae_baseline, mse_baseline = baseline_errors.loc[group]

      # Compute percentage difference
      mae_diff = ((mae_model - mae_baseline) / mae_baseline) * 100
      mse_diff = ((mse_model - mse_baseline) / mse_baseline) * 100

      # Log results
      print(f"{group} ({label} Group):")
      print(f"   Model  -> MAE: {int(round(mae_model))}, MSE: {int(round(mse_model))}")
      print(f"   Baseline -> MAE: {int(round(mae_baseline))}, MSE: {int(round(mse_baseline))}")
      print(f"   Difference -> MAE: {mae_diff:.2f}%, MSE: {mse_diff:.2f}% {'(Worse)' if mae_diff > 0 else '(Better)'}\n")

      # Store results for CSV export
      self.evaluation_results.append([group, label, round(mae_model, 2), round(mse_model, 2), round(mae_baseline, 2), round(mse_baseline, 2)])

  def export_evaluation_to_csv(self):
    # Ensure the evaluations directory exists
    evaluations_dir = "evaluations"
    os.makedirs(evaluations_dir, exist_ok=True)

    # Define CSV path
    csv_path = os.path.join(evaluations_dir, f"evaluation_{self.season}_steps_{self.steps}_prev_season_{self.include_prev_season}.csv")

    # Create DataFrame
    df = pd.DataFrame(self.evaluation_results, columns=["Group", "Metric", "Model MAE", "Model MSE", "Baseline MAE", "Baseline MSE"])

    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"Evaluation results saved to {csv_path}")

  def _load_predictions(self):
    directory = f'predictions/{self.model_type.value}/steps_{self.steps}_prev_season_{self.include_prev_season}/gws/{self.season}'
    
    if os.path.exists(directory):
      predictions_df = pd.DataFrame()

      path = Path(directory)
      for csv_file in path.glob("*.csv"):
        df = pd.read_csv(csv_file)
        predictions_df = pd.concat([predictions_df, df], ignore_index=True)
      
      return predictions_df
    else:
      raise Exception(f"Predictions for {season}, {steps} steps and prev season {include_prev_season} not found")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run the model evaluation.')
  parser.add_argument('--season', type=str, nargs='?', default='2024-25', help='Season to evaluate in the format 20xx-yy')
  parser.add_argument('--steps', type=int, nargs='?', default=5, help='Steps to evaluate')
  parser.add_argument('--prev_season', action='store_true', help='Set this flag to include prev season data. Defaults to false.')
  parser.add_argument('--model', type=str, help='The model to use', choices=[m.value for m in ModelType])
  
  args = parser.parse_args()

  season = args.season
  steps = args.steps
  include_prev_season = args.prev_season

  try:
    model_type = ModelType(args.model)
  except ValueError:
    print(f"Error: Invalid model type '{args.model}'. Choose from {', '.join(m.value for m in ModelType)}")
    exit(1)
    
  evaluate = Evaluate(
    season, 
    steps,
    model_type,
    include_prev_season)
  evaluate.evaluate_model()
