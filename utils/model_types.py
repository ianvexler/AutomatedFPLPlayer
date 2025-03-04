from enum import Enum

class ModelType(Enum):
  RANDOM_FOREST = 'random_forest'
  ADABOOST = 'adaboost'
  GRADIENT_BOOST = 'gradient_boost'
  XGBOOST = 'xgboost'
  ML_LSTM = 'ml_lstm'
  LSTM = 'lstm'