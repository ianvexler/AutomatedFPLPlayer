import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer
import joblib
import os
from utils.feature_selector import FeatureSelector
from utils.model_types import ModelType

class FeatureScaler:
  def __init__(self, position, model_type):
    """Initialize the scaler with a position and optional thresholds."""
    self.position = position
    self.scalers = {}  # Dictionary to store scalers per feature

    self.NO_SCALE = ['gw_decay']

    self.model_type = model_type
    self.directory = f"models/{self.model_type.value}/scalers"
    self.feature_selector = FeatureSelector()

  def fit(self, data, features):
    """Fit the selected scalers for each feature."""
    features = list(filter(lambda x: x not in self.NO_SCALE, features))
    
    for feature in data[features].columns:
      # TODO: Maybe use different scalers for different features
      self.scalers[feature] = RobustScaler()

      self.scalers[feature].fit(data[[feature]])
      self.save_scaler(feature)

    # Fits target column using only data from GWs > 1
    target = self.feature_selector.TARGET
    target_data = data[data['GW'] >= 1][target]

    # Initialize and fit target scaler
    self.scalers['target'] = MinMaxScaler()
    self.scalers['target'].fit(target_data.to_frame())
    self.save_scaler('target')

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

  def save_scaler(self, feature):
    """Save the scaler to disk for a specific feature."""
    os.makedirs(self.directory, exist_ok=True)

    scaler_path = f'{self.directory}/scaler_{self.position}_{feature}.pkl'
    joblib.dump(self.scalers[feature], scaler_path)

  def load_scalers(self, columns):
    """Load the scalers from disk for each feature."""
    for feature in columns:
      if feature in self.NO_SCALE:
        continue

      self.scalers[feature] = joblib.load(f'{self.directory}/scaler_{self.position}_{feature}.pkl')
    self.scalers['target'] = joblib.load(f'{self.directory}/scaler_{self.position}_target.pkl')
