import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from utils.feature_selector import FeatureSelector
from utils.feature_scaler import FeatureScaler

class LSTMModel:
  def __init__(
    self, 
    time_steps, 
    position,
    features
  ):
    self.time_steps = time_steps
    self.position = position
    self.features = features

  def build_model(self):
    # Model Architecture (Single-Layer LSTM)
    input_layer = Input(shape=(self.time_steps, len(self.features)))

    x = LSTM(32, kernel_regularizer=l2(0.001))(input_layer)  # Lower L2 regularization
    x = Dropout(0.3)(x)  # Moderate dropout

    output_layer = Dense(1, activation='linear', name=f"{self.position}_output")(x)
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile Model with a Lower Learning Rate
    model.compile(optimizer=Adam(learning_rate=3e-4), loss='mean_squared_error', metrics=['mae'])

    return model
