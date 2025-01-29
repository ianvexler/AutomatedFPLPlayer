from enum import Enum

class ModelType(Enum):
  RANDOM_FOREST = 'random-forest'
  ADABOOST = 'adaboost'
  GRADIENT_BOOST = 'gradient-boost'
  XGBOOST = 'xgboost'
  BI_LSTM = 'bi-lstm'
  LSTM = 'lstm'