from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from evaluate import Evaluate
from sklearn.metrics import mean_squared_error
from utils.ModelTypes import ModelType

TARGET = 'total_points'

class Model:
  def __init__(self, data, model = ModelType.RANDOM_FOREST):
    self.data = data

    self.evaluate = Evaluate()

    self.model = self.__set_model(model)
  
  # TODO: add more models
  def __set_model(self, model: ModelType):
    match model:
      case ModelType.RANDOM_FOREST:
        return RandomForestRegressor()

  def predict(self):
    x_train, x_test, y_train, y_test = self.get_train_test_data(self.data)

    self.model.fit(x_train, y_train)

    predictions = self.model.predict(x_test)

    # Maybe move?
    score = self.model.score(x_test, y_test)
    mse = mean_squared_error(y_test, predictions)

    print(f'R^2 score: {score}')
    print(f'Mean Squared Error: {mse}')

    test_indexes = x_test.index

    results_df = self.evaluate.format_results(test_indexes, predictions, y_test)
    return results_df

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