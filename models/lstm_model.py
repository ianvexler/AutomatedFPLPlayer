import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from utils.feature_selector import FeatureSelector
from utils.feature_scaler import FeatureScaler

class LSTMModel:
  def __init__(self, time_steps, position, include_prev_season, include_fbref, multi_layer=False):
    self.time_steps = time_steps
    self.position = position
    self.include_prev_season = include_prev_season
    self.include_fbref = include_fbref
    self.multi_layer = multi_layer

  def build_model(self):
    feature_selector = FeatureSelector()
    features = feature_selector.get_features_for_position(
        self.position, include_prev_season=self.include_prev_season, include_fbref=self.include_fbref
    )

    input_layer = Input(shape=(self.time_steps, len(features)))

    if self.multi_layer:
      x = LSTM(32, return_sequences=True, kernel_regularizer=l2(0.005))(input_layer)
      x = Dropout(0.3)(x)  # Reduced from 0.4
      x = LSTM(16, kernel_regularizer=l2(0.005))(x)
      x = Dropout(0.2)(x)  # Reduced from 0.3
    else:
      x = LSTM(32, kernel_regularizer=l2(0.005))(input_layer)
      x = Dropout(0.3)(x)  # Reduced from 0.4

    output_layer = Dense(1, activation='linear', name=f"{self.position}_output")(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    # Use a lower learning rate for smoother training
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error', metrics=['mae'])

    return model
