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

class DropoutEvaluation:
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
    dropouts_to_test = [0.2, 0.3, 0.4, 0.5]
    model_type = ModelType.LSTM
    results = []

    for dropout in dropouts_to_test:
      print(f"Training model with dropout rate {dropout}...")
      lstm_config = { 'units': 64, 'dropout': dropout }

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
          'dropout': dropout,
          'val_rmse': val_rmse,
          'val_mae': val_mae
        })

    df = pd.DataFrame(results)
    print("\nDropout Evaluation Results:\n")
    print(tabulate(df, headers='keys', tablefmt='pretty', floatfmt=".4f"))

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

  model_evaluation = DropoutEvaluation(
    season=args.season,
    time_steps=args.steps,
    include_prev_season=args.prev_season,
    include_fbref=args.fbref,
    include_season_aggs=args.season_aggs,
    include_teams=args.teams,
    gw_lamda_decay=args.gw_decay,
    n_features=args.top_features)

  model_evaluation.evaluate()