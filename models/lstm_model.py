import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from utils.feature_selector import FeatureSelector
from utils.feature_scaler import FeatureScaler
from utils.model_types import ModelType

class LSTMModel:
  def __init__(
    self, 
    time_steps, 
    position,
    features,
    model_type,
    config=None
  ):
    self.time_steps = time_steps
    self.position = position
    self.features = features
    self.model_type = model_type
    self.config = config

    self.units = config['units'] if config else 32
    self.dropout = config['dropout'] if config else 0.3
    self.learning_rate = config['learning_rate'] if config else 3e-4

  def build_model(self):
    input_layer = Input(shape=(self.time_steps, len(self.features)))

    if self.model_type.value == ModelType.LSTM.value:
      x = LSTM(
        self.units, kernel_regularizer=l2(0.001)
      )(input_layer)
    
    elif self.model_type.value == ModelType.BI_LSTM.value:
      x = Bidirectional(
        LSTM(self.units, kernel_regularizer=l2(0.001))
      )(input_layer)
    
    elif self.model_type.value == ModelType.ML_LSTM.value:
      x = LSTM(self.units, return_sequences=True)(input_layer)
      x = LSTM(int(self.units / 2), kernel_regularizer=l2(0.001))(x)
    
    else:
      raise Exception('Invalid LSTM model selected')

    x = Dropout(self.dropout)(x)

    output_layer = Dense(1, activation='linear', name=f"{self.position}_output")(x)
    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error', metrics=['mae'])

    return model
