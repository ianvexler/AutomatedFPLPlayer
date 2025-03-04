import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from utils.feature_selector import FeatureSelector
from utils.feature_scaler import FeatureScaler

class LSTMModel:
  def __init__(self, time_steps, position, include_prev_season, multi_layer=False):
    self.time_steps = time_steps
    self.position = position
    self.include_prev_season = include_prev_season

  def build_model(self):
    feature_selector = FeatureSelector()
    features = feature_selector.get_features_for_position(self.position, self.include_prev_season)
    
    # Define the input layer
    input_layer = Input(shape=(self.time_steps, len(features)))

    if self.multi_layer:
      # Multi-Layer LSTM
      x = LSTM(64, return_sequences=True)(input_layer)
      x = Dropout(0.5)(x)
      x = LSTM(32)(x)
    else:
      # Single-Layer LSTM
      x = LSTM(64)(input_layer)

    # Single output layer for position-specific targets
    output_layer = Dense(1, activation='linear', name=f"{self.position}_output")(x)

    # Create the model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

    return model
