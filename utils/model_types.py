from enum import Enum

class ModelType(Enum):
  RANDOM_FOREST = 'random_forest'
  ADABOOST = 'adaboost'
  GRADIENT_BOOST = 'gradient_boost'
  XGBOOST = 'xgboost'
  LSTM = 'lstm'
  ML_LSTM = 'ml_lstm'
  BI_LSTM = 'bi_lstm'