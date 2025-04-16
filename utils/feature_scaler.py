import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer
import joblib
import os
from utils.feature_selector import FeatureSelector

class FeatureScaler:
  def __init__(self, position):
    """Initialize the scaler with a position and optional thresholds."""
    self.position = position
    self.scalers = {}  # Dictionary to store scalers per feature

    self.NO_SCALE = ['gw_decay']

    self.feature_selector = FeatureSelector()

  def fit(self, data, features):
    """Fit the selected scalers for each feature."""
    features = list(filter(lambda x: x not in self.NO_SCALE, features))
    
    for feature in data[features].columns:
      # TODO: Maybe use different scalers for different features
      self.scalers[feature] = RobustScaler()

      self.scalers[feature].fit(data[[feature]])

    # Fits target column using only data from GWs > 1
    target = self.feature_selector.TARGET
    target_data = data[data['GW'] >= 1][target]

    # Initialize and fit target scaler
    self.scalers['target'] = MinMaxScaler()
    self.scalers['target'].fit(target_data.to_frame())

  def transform_data(self, data):
    """Apply the scalers to transform the dataset."""
    transformed_data = data.copy()

    for feature in data.columns:
      if feature in self.NO_SCALE:
        transformed_data[feature] = data[[feature]]
        continue
      
      transformed_data[feature] = self.scalers[feature].transform(data[[feature]])
    return transformed_data

  def transform(self, value, feature, target=False):
    feature_key = 'target' if target else feature

    value_df = pd.DataFrame([[value]], columns=[feature])
    return self.scalers[feature_key].transform(value_df)[0][0]  

  def inverse_transform(self, value_df, feature_key):
    """Apply the inverse transformation to return to the original scale for the passed features."""
    return self.scalers[feature_key].inverse_transform(value_df)
