from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error
from utils.model_types import ModelType
# from xgboost import XGBRegressor
from models.lstm_model import LSTMModel

TARGET = 'total_points'

class Model:
  def __init__(self, data, train_data, model = ModelType.RANDOM_FOREST):
    self.data = data
    self.train_data = train_data

    self.model = self.__set_model(model)
  
  def __set_model(self, model: ModelType):
    match model:
      case ModelType.RANDOM_FOREST:
        return RandomForestRegressor()
      case ModelType.ADABOOST:
        return AdaBoostRegressor()
      case ModelType.GRADIENT_BOOST:
        return GradientBoostingRegressor()
      # case ModelType.XGBOOST:
      #   return XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
      case ModelType.LSTM():
        return LSTMModel()

  """
  Trains the model on the training data and makes predictions on the full data,
  using only the features that appear in both datasets.

  Returns:
    indexes: TODO
    predictions: TODO
    targets: TODO
  """
  def predict(self):
    x_train, y_train = self.split_data(self.train_data)

    x_full, y_full = self.split_data(self.data)

    # Identify common features between training and full datasets
    common_features = x_train.columns.intersection(x_full.columns)

    # Filter both datasets to use only common features
    x_train = x_train[common_features]
    x_full = x_full[common_features]

    # Fit the model using the filtered training data
    self.model.fit(x_train, y_train)

    # Make predictions using the filtered full data
    predictions = self.model.predict(x_full)

    return x_full.index, predictions, y_full

  def test_predict(self):
    x_train, x_test, y_train, y_test = self.split_train_test_data(self.train_data)

    self.model.fit(x_train, y_train)
    predictions = self.model.predict(x_test)

    return x_test.index, predictions, y_test

  """
  Splits the data into training and testing data

  Params:
    data_df: Pd Data Frame with the formatted data

  Returns:
    x_train: Features training data
    x_test: Features test data
    y_train: Target training data
    y_test: Target test data
  """
  def split_train_test_data(self, data_df: pd.DataFrame):
    x = data_df.loc[:, data_df.columns != TARGET]
    y = data_df[TARGET]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test

  """
  Splits the data into features (X) and labels (y).

  Params:
    data_df: pandas DataFrame with the formatted data

  Returns:
    x: Features data
    y: Target data
  """
  def split_data(self, data_df: pd.DataFrame):
    x = data_df.loc[:, data_df.columns != TARGET]
    y = data_df[TARGET]

    return x, y