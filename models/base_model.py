import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

TARGET_COLUMN = 'total_points'

class BaseModel:
  def __init__(self, data, time_steps=7):
    self.data = data.sort_values(by=['id', 'GW'])
    self.time_steps = time_steps

  def _calculate_form(self, window=3):
    """Calculate form as a rolling average of total points."""
    self.data['form'] = self.data.groupby('id')[TARGET_COLUMN].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)

  def _prepare_features(self, gw):
    """Prepare features and targets for a given gameweek."""
    train_data = self.data[self.data['GW'] < gw].copy()
    test_data = self.data[self.data['GW'] == gw].copy()

    # Feature engineering: add lag features and rolling averages
    for lag in range(1, self.time_steps + 1):
      train_data[f'lag_{lag}'] = train_data.groupby('id')[TARGET_COLUMN].shift(lag)
      test_data[f'lag_{lag}'] = test_data.groupby('id')[TARGET_COLUMN].shift(lag)
    
    # Drop rows with NaN values from lag features
    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    # Define features and target
    feature_columns = [col for col in train_data.columns if col not in ['id', 'GW', TARGET_COLUMN]]
    X_train = train_data[feature_columns]
    y_train = train_data[TARGET_COLUMN]
    X_test = test_data[feature_columns]
    y_test = test_data[TARGET_COLUMN]

    return X_train, y_train, X_test, y_test

  def train(self):
    """This method should be overridden by subclasses to implement model-specific logic."""
    raise NotImplementedError("Subclasses should implement this method.")

  def predict(self):
    """This method should be overridden by subclasses to implement model-specific logic."""
    raise NotImplementedError("Subclasses should implement this method.")

