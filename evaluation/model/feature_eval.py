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

class FeatureEvaluation:
  def __init__(
    self, 
    season, 
    time_steps, 
    model_type, 
    include_prev_season=False, 
    include_fbref=False, 
    include_season_aggs=False, 
    include_teams=False,
    gw_lamda_decay=0.02,
    n_features=20
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
    
    self.n_features = n_features

    self.model = None

    self.FILE_NAME = f"steps_{self.time_steps}_prev_season_{self.include_prev_season}_fbref_{self.include_fbref}_season_aggs_{self.include_season_aggs}_teams_{self.include_teams}_gw_decay_{self.gw_lamda_decay}"
    
  def evaluate(self):
    self.model = Model(
      season=args.season,
      time_steps=args.steps,
      include_prev_season=args.prev_season,
      model_type=model_type,
      include_fbref=args.fbref,
      include_season_aggs=args.season_aggs,
      include_teams=args.teams,
      gw_lamda_decay=args.gw_decay)

    training_data, _ = self.model.train()

    for position in ['GK', 'DEF', 'MID', 'FWD']:
      X, y = training_data[position]
      
      # Compute full importance dict once
      importance_dict = self._get_permutation_importance(
        position=position,
        X=X,
        y=y)

      for top_n in [10, 15, 20, 25, 30]:
        top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        feature_names = [name for name, _ in top_features]
        print(f"{position} Top {top_n}: {feature_names}")

        self._plot_permutation_importance(
          importance_dict,
          position=position,
          save_path=f"plots/permutation_importance/test2/{top_n}/{self.FILE_NAME}",
          top_n=top_n
        )

  def _permutation_importance_lstm(self, model, X_val, y_val, feature_names, scaler, n_repeats=5):
    y_val_original = y_val.flatten()
    predictions_baseline = model.predict(X_val)
    
    predictions_baseline = scaler.inverse_transform(
      pd.DataFrame(predictions_baseline, columns=['target']), 'target'
    ).flatten()
    
    baseline_rmse = np.sqrt(mean_squared_error(y_val_original, predictions_baseline))
    importances = []

    for i in range(X_val.shape[2]):
      scores = []
      for _ in range(n_repeats):
        X_permuted = X_val.copy()
        np.random.shuffle(X_permuted[:, :, i])
        y_pred_permuted = model.predict(X_permuted)
        
        y_pred_permuted = scaler.inverse_transform(
          pd.DataFrame(y_pred_permuted, columns=['target']), 'target'
        ).flatten()
        
        rmse = np.sqrt(mean_squared_error(y_val_original, y_pred_permuted))
        scores.append(rmse)
      
      importances.append(np.mean(scores) - baseline_rmse)

    return dict(sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True))

  def _permutation_importance_tree(self, model, X_val, y_val, feature_names, n_repeats=5):
    baseline = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))

    importances = []
    for i in range(X_val.shape[1]):
      scores = []
      for _ in range(n_repeats):
        X_permuted = X_val.copy()
        np.random.shuffle(X_permuted[:, i])
        y_pred = model.predict(X_permuted)
        
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        scores.append(rmse)
      importances.append(np.mean(scores) - baseline)

    return dict(sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True))

  def _plot_permutation_importance(self, importance_dict, position=None, save_path=None, top_n=20):
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, importances = zip(*sorted_items)

    plt.figure(figsize=(10, max(4, len(features) * 0.4)))
    sns.barplot(x=importances, y=features, palette='viridis')
    plt.xlabel('Change in RMSE')
    plt.title(f'Permutation Feature Importance{f" - {position}" if position else ""}')
    plt.tight_layout()

    if save_path:
      os.makedirs(save_path, exist_ok=True)
      filename = f"perm_importance_{position}.png" if position else "perm_importance.png"
      full_path = os.path.join(save_path, filename)
      plt.savefig(full_path, bbox_inches='tight')
    else:
      plt.show()


  def _get_permutation_importance(self, position, X, y):
    if self._is_model_sequential():
      y = self.model.scalers[position].inverse_transform(
        pd.DataFrame(y, columns=['target']), 'target'
      ).flatten()
    else:
      y = y.flatten()

    y = np.array(y).flatten()
    X = np.array(X)

    high_var_mask = y > 2

    X_high_var = X[high_var_mask]
    y_high_var = y[high_var_mask]

    if not self._is_model_sequential():
      X = X.reshape(X.shape[0], -1)
    
    feature_names = FeatureSelector().get_features_for_position(
      position, 
      include_prev_season=self.include_prev_season, 
      include_fbref=self.include_fbref, 
      include_season_aggs=self.include_season_aggs,
      include_teams=self.include_teams
    )

    print(f"Total {position} Feature Count: {len(feature_names)}")

    if self._is_model_sequential():
      importance_dict = self._permutation_importance_lstm(
        self.model.models[position], X, y, feature_names, 
        self.model.scalers[position]
      )
    else:
      importance_dict = self._permutation_importance_tree(
        self.model.models[position], X, y, feature_names
      )

    return importance_dict

  def _is_model_sequential(self):
    return self.model_type in { ModelType.LSTM, ModelType.ML_LSTM, ModelType.BI_LSTM }

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Run the model with optional training.')
  parser.add_argument('--steps', type=int, nargs='?', const=5, default=5, help='Time step for data window. Defaults to 7 if not provided or null.')
  parser.add_argument('--season', type=str, nargs='?', default='2023-24', help='Season to simulate in the format 20xx-yy.')
  parser.add_argument('--prev_season', action='store_true', help='Set this flag to include prev season data. Defaults to false.')
  parser.add_argument('--model', type=str, help='The model to use', choices=[m.value for m in ModelType])
  parser.add_argument('--fbref', action='store_true', help='Include FBref data.')
  parser.add_argument('--season_aggs', action='store_true', help='Include season aggregate data.')
  parser.add_argument('--teams', action='store_true', help='Include teams data.')
  parser.add_argument('--gw_decay', type=float, default=0.02, help='The lambda decay applied in gw decay')

  parser.add_argument('--n_features', type=int, default=None)

  args = parser.parse_args()

  try:
    model_type = ModelType(args.model)
  except ValueError:
    print(f"Error: Invalid model type '{args.model}'. Choose from {', '.join(m.value for m in ModelType)}")
    exit(1)

  model_evaluation = FeatureEvaluation(
    season=args.season,
    time_steps=args.steps,
    include_prev_season=args.prev_season,
    model_type=model_type,
    include_fbref=args.fbref,
    include_season_aggs=args.season_aggs,
    include_teams=args.teams,
    gw_lamda_decay=args.gw_decay,
    n_features=args.n_features)
  model_evaluation.evaluate()