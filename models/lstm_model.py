import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from utils.feature_selector import FeatureSelector
from utils.feature_scaler import FeatureScaler

class LSTMModel:
  def __init__(self, gw_data, teams_data, fixtures, players_data, season, time_steps=7, train=False):
    self.gw_data = gw_data.sort_values(by=['id', 'kickoff_time'])  # Gameweek-level data
    self.teams_data = teams_data
    self.fixtures = fixtures
    self.players_data = players_data
    self.season = season
    self.feature_selector = FeatureSelector()

    self.target = self.feature_selector.TARGET

    self.time_steps = time_steps

    # Selelects the position for all features
    for position in ['GK', 'DEF', 'MID', 'FWD']:  
      position_gw_data = self._filter_data_by_position(position)

    self.models = {
      'GK': self._build_model('GK'),
      'DEF': self._build_model('DEF'),
      'MID': self._build_model('MID'),
      'FWD': self._build_model('FWD')
    }

    self.scalers = {
      'GK': FeatureScaler('GK'),
      'DEF': FeatureScaler('DEF'),
      'MID': FeatureScaler('MID'),
      'FWD': FeatureScaler('FWD')
    }

    if train:
      for position in self.models:
        self.train(position)
    else:
      for position in self.models:
        directory = 'models/trained'

        self.models[position] = tf.keras.models.load_model(os.path.join(directory, f'lstm_model_{position}.keras'))
        
        # Improve
        features = self.feature_selector.get_features_for_position(position)
        
        scaler = self.scalers[position]
        scaler.load_scalers(features)

  def _build_model(self, position):
    features = self.feature_selector.get_features_for_position(position)

    # Define the input layer
    input_layer = Input(shape=(self.time_steps, len(features)))

    # Shared LSTM layers
    shared_lstm = LSTM(64, return_sequences=True)(input_layer)
    shared_lstm = Dropout(0.5)(shared_lstm)
    shared_lstm = LSTM(32)(shared_lstm)

    # Single output layer for position-specific targets
    output_layer = Dense(1, activation='linear', name=f"{position}_output")(shared_lstm)

    # Create the model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

    return model

  def train(self, position):
    # Prepare sequences from historical gameweek data
    X_reshaped, y_reshaped, _ = self._prepare_sequences(position)

    # Train the model
    model = self.models[position]

    history = model.fit(X_reshaped, y_reshaped, epochs=80, batch_size=32, validation_split=0.2, verbose=1)

    directory = 'models/trained'
    os.makedirs(directory, exist_ok=True)

    model.save(os.path.join(directory, f'lstm_model_{position}.keras'))

    print(f"Saved model and scalers for {position}\n")

  def _prepare_sequences(self, position):
    X_reshaped = []
    y_reshaped = []
    player_ids = []  # To keep track of player IDs for each sequence

    # Group data by player to generate sequences per player and orders them

    position_gw_data = self._filter_data_by_position(position)
    # position_gw_data = self.feature_selector.reorder_features(position_gw_data, position)

    features = self.feature_selector.get_features_for_position(position)

    feature_scaler = self._fit_scaler(position_gw_data, features, position)

    scaled_data = position_gw_data.copy()
    scaled_data[features] = feature_scaler.transform(position_gw_data[features])

    for player_id, player_data in scaled_data.groupby('id'):
      player_data = player_data.reset_index(drop=True)

      # Use all feature columns for input
      X = player_data[features].values
      
      # Use only target columns for prediction targets
      y = player_data[self.target].values

      # Use sliding window to create sequences
      for i in range(len(player_data) - self.time_steps):
        X_sequence = X[i:i + self.time_steps]  # Sequence of past data (inputs)
        y_target = y[i + self.time_steps]  # Predict stats for the next gameweek (target)

        X_reshaped.append(X_sequence)
        y_reshaped.append(y_target)
        player_ids.append(player_id)  # Keep track of player IDs

    return np.array(X_reshaped), np.array(y_reshaped), player_ids

  def _fit_scaler(self, position_gw_data, columns, position):
    scaler = self.scalers[position]
    data = position_gw_data[columns]
    scaler.fit(data)  # Fit the scaler on all columns (features and targets)

    return scaler

  """
  Filters the gw_data DataFrame to include only rows where the player's position matches the specified position.

  Args:
  position (str): The position to filter by (e.g., 'Forward', 'Midfielder').

  Returns:
  pandas.DataFrame: A DataFrame containing only the rows where the player's position matches the specified position.
  """
  def _filter_data_by_position(self, position):
    return self.gw_data[self.gw_data['position'] == position]

  def predict_season(self):
    print(f"Started Season Prediction")

    # Initialize a dictionary to accumulate predictions for each player
    player_aggregates = {}

    for gw, gw_group in self.fixtures.groupby('GW'):
      current_gw = gw
      current_data = self.gw_data[self.gw_data['GW'] <= current_gw]

      gw_aggregates = {}
      for _, game in gw_group.iterrows():
        home_team_id = game['team_h']
        away_team_id = game['team_a']

        players_data = current_data[
          ((current_data['team'] == home_team_id) | (current_data['team'] == away_team_id)) & 
          (current_data['GW'] < current_gw)
        ]
        
        # Predict and accumulate results for players in both teams
        predictions = self._predict_players_game(players_data, current_gw)

        players_gw_data = current_data[
          ((current_data['team'] == home_team_id) | (current_data['team'] == away_team_id)) & 
          (current_data['GW'] == current_gw)
        ]
        self._format_and_save_match_prediction(predictions, home_team_id, away_team_id, players_gw_data, current_gw)

        # Add to aggregate
        for player_id, prediction in predictions.items():
          player_aggregates.setdefault(player_id, 0)
          player_aggregates[player_id] += prediction

          gw_aggregates.setdefault(player_id, 0)
          gw_aggregates[player_id] += prediction

      players_data = current_data[current_data['GW'] == current_gw]
      self._format_and_save_gw_prediction(gw_aggregates, players_data, current_gw)

    # Convert the accumulated results to a DataFrame
    aggregate_predictions = []
    for player_id, metrics in player_aggregates.items():
      aggregate_predictions.append({
        'player_id': player_id,
        **metrics
      })

    # Convert to DataFrame and save as CSV
    predictions_df = self._format_and_save_predictions(aggregate_predictions)

    return predictions_df

  def _predict_players_game(self, players_data, current_gw):
    predictions = {}

    players_data.to_csv('jaksldfkjlk.csv')

    for player_id, player_data in players_data.groupby('id'):
      position = self._get_player_position(player_id)

      player_sequence = self._get_player_sequence(player_data, position)
      
      combined_sequence = player_sequence[np.newaxis, :, :]

      # Returns unscaled prediction
      prediction = self._predict_player_performance(combined_sequence, position)

      predictions.setdefault(player_id, 0)
      predictions[player_id] += prediction
    
    return predictions

  # Returns total predicted points
  def _predict_player_performance(self, combined_sequence, position):
    prediction = self.models[position].predict(combined_sequence)    
    prediction = np.maximum(prediction, 0)

    prediction_df = pd.DataFrame(prediction, columns=[self.target])

    scaler = self.scalers[position]
    prediction_unscaled = scaler.inverse_transform(prediction_df)

    return prediction_unscaled.iloc[0][self.target]

  def _get_player_sequence(self, player_data, position):
    player_data = player_data.sort_values(by='GW').tail(self.time_steps)

    features = self.feature_selector.get_features_for_position(position)

    scaler = self.scalers[position]
    scaled_sequence = scaler.transform(player_data[features])

    return scaled_sequence.values

  def _get_player_position(self, player_id):
    position = self.players_data[self.players_data['id'] == player_id].iloc[0]['position']

    if position == None:
      raise Exception('Position could not be obtained for id: {player_id}')

    return position
  
  def _format_and_save_match_prediction(self, predictions, home_team_id, away_team_id, gw_data, current_gw):
    directory = f"predictions/matches/{self.season}"
    os.makedirs(directory, exist_ok=True)

    home_team = self.teams_data[self.teams_data['id'] == home_team_id].iloc[0]['name']
    away_team = self.teams_data[self.teams_data['id'] == away_team_id].iloc[0]['name']

    file_path = os.path.join(directory, f"GW{int(current_gw)}_{home_team}_vs_{away_team}.csv")

    predictions_df = pd.DataFrame.from_dict(predictions, orient='index', columns=['expected_points'])

    gw_predictions_df = gw_data.merge(predictions_df, left_on='id', right_index=True, how='left')
    gw_predictions_df['expected_points'] = gw_predictions_df['expected_points'].fillna(0)

    gw_predictions_df.to_csv(file_path, index=False)
    print(f"{home_team} vs {away_team} saved to {file_path}\n")
    
    return predictions

  def _format_and_save_gw_prediction(self, predictions, gw_data, current_gw):
    directory = f"predictions/gws/{self.season}"
    os.makedirs(directory, exist_ok=True)

    file_path = os.path.join(directory, f"GW{int(current_gw)}.csv")
    predictions_df = pd.DataFrame.from_dict(predictions, orient='index', columns=['expected_points'])

    gw_predictions_df = gw_data.merge(predictions_df, left_on='id', right_index=True, how='left')
    gw_predictions_df.to_csv(file_path, index=False)
    
    print(f"GW{int(current_gw)} saved to {file_path}\n")
    
    return predictions

  def _format_and_save_predictions(self, aggregate_predictions):
    # Convert aggregate predictions to a DataFrame
    predictions_df = pd.DataFrame(aggregate_predictions)

    directory = 'predictions'
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"predictions_{self.season}.csv")

    predictions_df.to_csv(file_path, index=False)
    print(f"Predictions saved to {file_path}")

    return predictions_df
