import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer
import joblib
import os
from utils.feature_selector import FeatureSelector

class FeatureScaler:
  def __init__(self, position, skew_threshold=1, outlier_threshold=1.5, feature_range=(0, 1)):
    """Initialize the scaler with a position and optional thresholds."""
    self.position = position
    self.scalers = {}  # Dictionary to store scalers per feature
    self.skew_threshold = skew_threshold
    self.outlier_threshold = outlier_threshold
    self.feature_range = feature_range

    self.directory = 'models/lstm/scalers'

  def get_feature_statistics(self, data):
    """Generate basic statistics for each feature (column) to guide scaler selection."""
    stats = {}
    stats['min'] = data.min()
    stats['max'] = data.max()
    stats['mean'] = data.mean()
    stats['std'] = data.std()
    stats['skew'] = data.skew()
    stats['iqr'] = data.quantile(0.75) - data.quantile(0.25)
    return stats

  def select_scaler(self, stats, features):
    """Automatically assign a scaler to each feature based on its statistics."""
    for feature in features:
      self.scalers[feature] = RobustScaler()
      continue

      # Extract scalar values from the pandas Series
      max_val = stats['max'][feature].max()
      min_val = stats['min'][feature].min()
      skew_val = abs(stats['skew'][feature].max())
      iqr_val = stats['iqr'][feature].max()
      std_val = stats['std'][feature].max()

      # Apply the scaler based on feature statistics
      if max_val <= 90 and min_val >= 0:
        print(f'{feature}: min max')
        self.scalers[feature] = MinMaxScaler(feature_range=(0, 90))
      elif skew_val > self.skew_threshold:
        print(f'{feature}: PowerTransformer')
        self.scalers[feature] = PowerTransformer(method='yeo-johnson')
      elif iqr_val > std_val * self.outlier_threshold:
        print(f'{feature}: robust')
        self.scalers[feature] = RobustScaler()
      else:
        print(f'{feature}: standard')
        self.scalers[feature] = StandardScaler()

  def fit(self, data):
    """Fit the selected scalers for each feature."""
    stats = self.get_feature_statistics(data)
    self.select_scaler(stats, data.columns)
    
    for feature in data.columns:
      self.scalers[feature].fit(data[[feature]])
      self.save_scaler(feature)

  def transform(self, data):
    """Apply the scalers to transform the dataset."""
    transformed_data = data.copy()

    for feature in data.columns:
      transformed_data[feature] = self.scalers[feature].transform(data[[feature]])

    return transformed_data

  def inverse_transform(self, data):
    """Apply the inverse transformation to return to the original scale for the passed features."""
    inverse_data = data.copy()
    
    for feature in data.columns:
      if feature in self.scalers:
        inverse_data[feature] = self.scalers[feature].inverse_transform(data[[feature]])
  
    return inverse_data

  def save_scaler(self, feature):
    """Save the scaler to disk for a specific feature."""
    os.makedirs(self.directory, exist_ok=True)

    scaler_path = f'{self.directory}/scaler_{self.position}_{feature}.pkl'
    joblib.dump(self.scalers[feature], scaler_path)

  def load_scalers(self, columns):
    """Load the scalers from disk for each feature."""

    for feature in columns:
      self.scalers[feature] = joblib.load(f'{self.directory}/scaler_{self.position}_{feature}.pkl')

