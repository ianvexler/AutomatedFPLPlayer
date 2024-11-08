from data.vaastav.data_loader import DataLoader as Vaastav
from sklearn.model_selection import train_test_split
import pandas as pd

TARGET = 'total_points'

class DataLoader:
  def get_data(self, season):
    data_loader = Vaastav(season)
    data = data_loader.get_full_season_data()

    return data
  
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
  def get_train_test_data(self, data_df: pd.DataFrame):
    x = data_df.loc[:, data_df.columns != TARGET]
    y = data_df[TARGET]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test
