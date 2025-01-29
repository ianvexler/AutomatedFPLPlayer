import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils.features import Features

class LSTMModel:
  def __init__(self, gw_data, season_data, teams_data, fixtures, players_data, season, time_steps=7, train=False):
    self.gw_data = gw_data.sort_values(by=['id', 'kickoff_time'])  # Gameweek-level data
    self.season_data = season_data  # Overall season stats, TODO
    self.teams_data = teams_data
    self.fixtures = fixtures
    self.players_data = players_data
    self.season = season

    self.time_steps = time_steps

    self.models = {
      'GK': self._build_model('GK'),
      'DEF': self._build_model('DEF'),
      'MID': self._build_model('MID'),
      'FWD': self._build_model('FWD')
    }

    self.scalers = {
      'GK': StandardScaler(),
      'DEF': StandardScaler(),
      'MID': StandardScaler(),
      'FWD': StandardScaler()
    }

    self.target_scalers = {
      'GK': StandardScaler(),
      'DEF': StandardScaler(),
      'MID': StandardScaler(),
      'FWD': StandardScaler()
    }

    if train:
      for position in self.models:
        self.train(position)
    else:
      for position in self.models:
        directory = 'models/trained'

        for position in self.models:
          self.models[position] = tf.keras.models.load_model(os.path.join(directory, f'lstm_model_{position}.keras'))
          self.scalers[position] = joblib.load(os.path.join(directory, f'scaler_{position}.pkl'))
          self.target_scalers[position] = joblib.load(os.path.join(directory, f'target_scaler_{position}.pkl'))

  def _build_model(self, position):
    features = Features.get_features_for_position(position)
    targets = Features.get_targets_for_position(position)

    model = Sequential()
    model.add(Input(shape=(self.time_steps, len(features))))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(32))
    model.add(Dense(len(targets)))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

  def train(self, position):
    # Prepare sequences from historical gameweek data
    X_reshaped, y_reshaped, _ = self._prepare_sequences(position)
    
    # Scale features
    feature_scaler = self.scalers[position]
    feature_scaler.fit(X_reshaped.reshape(-1, X_reshaped.shape[-1]))
    X_reshaped = feature_scaler.transform(X_reshaped.reshape(-1, X_reshaped.shape[-1])).reshape(X_reshaped.shape)

    # Scale targets
    target_scaler = self.target_scalers[position]
    target_scaler.fit(y_reshaped)

    # Train the model
    model = self.models[position]
    history = model.fit(X_reshaped, y_reshaped, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    directory = 'models/trained'
    os.makedirs(directory, exist_ok=True)

    model.save(os.path.join(directory, f'lstm_model_{position}.keras'))
    joblib.dump(feature_scaler, os.path.join(directory, f'scaler_{position}.pkl'))
    joblib.dump(target_scaler, os.path.join(directory, f'target_scaler_{position}.pkl'))

    print(f"Saved model and scaler for {position}\n")

  def _prepare_sequences(self, position):
    X_reshaped = []
    y_reshaped = []
    player_ids = []  # To keep track of player IDs for each sequence

    # Group data by player to generate sequences per player
    for player_id, player_data in self.gw_data.groupby('id'):
      player_position = self._get_player_position(player_id)
      if player_position != position:
        continue

      player_data = player_data.reset_index(drop=True)

      # Use all feature columns for input
      features = Features.get_features_for_position(position)
      X = player_data[features].values
      
      # Use only target columns for prediction targets
      targets = Features.get_targets_for_position(position)
      y = player_data[targets].values

      # Use sliding window to create sequences
      for i in range(len(player_data) - self.time_steps):
        X_sequence = X[i:i + self.time_steps]  # Sequence of past data (inputs)
        y_target = y[i + self.time_steps - 1]  # Predict stats for the next gameweek (target)

        X_reshaped.append(X_sequence)
        y_reshaped.append(y_target)
        player_ids.append(player_id)  # Keep track of player IDs

    return np.array(X_reshaped), np.array(y_reshaped), player_ids

  def predict_season(self):
    # Initialize a dictionary to accumulate predictions for each player
    player_aggregates = {}

    current_data = self._initialize_with_last_gws()
    current_data['kickoff_time'] = pd.to_datetime(current_data['kickoff_time'], errors='coerce')

    for _, game in self.fixtures.sort_values(by=['kickoff_time']).iterrows():
      home_team_id = game['team_h']
      away_team_id = game['team_a']
      kickoff_time = pd.to_datetime(game['kickoff_time'])

      home_team_context = self.teams_data[self.teams_data['id'] == home_team_id].iloc[0]
      away_team_context = self.teams_data[self.teams_data['id'] == away_team_id].iloc[0]

      # Format and combine team contexts
      home_team_context = [f'home_team_{col}' for col in Features.TEAM_FEATURES]
      away_team_context = [f'away_team_{col}' for col in Features.TEAM_FEATURES]
      combined_teams_context = home_team_context + away_team_context

      home_players = current_data[
        (current_data['team'] == home_team_id) &
        (current_data['kickoff_time'].dt.date <= kickoff_time.date())
      ]
      away_players = current_data[
        (current_data['team'] == away_team_id) & 
        (current_data['kickoff_time'].dt.date <= kickoff_time.date())
      ]

      # Predict and accumulate results for both teams
      home_players_predictions = self._predict_players_game(home_players, combined_teams_context, kickoff_time, player_aggregates)
      away_players_predictions = self._predict_players_game(away_players, combined_teams_context, kickoff_time, player_aggregates)

      predictions = np.concatenate((home_players_predictions, away_players_predictions), axis=0)

      current_data = self._update_data_with_predictions(predictions, current_data, kickoff_time)

    # Convert the accumulated results to a DataFrame
    aggregate_results = []
    for player_id, metrics in player_aggregates.items():
      aggregate_results.append({
        'player_id': player_id,
        **metrics
      })

    # Convert to DataFrame and save as CSV
    predictions_df = pd.DataFrame(aggregate_results)
    predictions_df.to_csv('predictions.csv', index=False)

    return predictions_df

  def _predict_players_game(self, players, teams_context, kickoff_time, player_aggregates):
    predictions = []

    for player_id, player_data in players.groupby('id'):
      position = self._get_player_position(player_id)
      targets = Features.get_targets_for_position(position)

      player_sequence = self._get_player_sequence(player_data, position)
      
      if (player_data['season'].iloc[0] == self.season):
        combined_sequence = self._combined_team_player_context(player_sequence, teams_context)
      else:
        combined_sequence = player_sequence[np.newaxis, :, :]

      scaler = self.scalers[position]
      scaled_sequence = scaler.transform(combined_sequence.reshape(-1, combined_sequence.shape[-1])).reshape(combined_sequence.shape)
      
      # Returns unscaled prediction
      prediction = self._predict_player_performance(scaled_sequence, position)

      predictions.append(self._create_prediction_entry(player_id, kickoff_time, prediction, position))

      if player_id not in player_aggregates:
        player_aggregates[player_id] = {metric: 0 for metric in targets}

      # Accumulate predictions for each metric
      for i, metric in enumerate(targets):
          player_aggregates[player_id][metric] += prediction[0][i]

    return predictions

  # Add team context to the player's last time step in the sequence
  def _combined_team_player_context(self, player_sequence, teams_sequence):
    team_features = teams_sequence[Features.GW_TEAM_FEATURES].values
    combined_sequence = player_sequence.copy()

    combined_sequence[-1] = np.concatenate([combined_sequence[-1], team_features])

    return combined_sequence[np.newaxis, :]

  def _get_team_sequence(self, current_data, team_id):
    team_data = current_data[(current_data['id'] == team_id)].sort_values(by='kickoff_time').tail(self.time_steps)

    sequence = team_data[Features.get_features_for_position(position)].values
    return sequence

  def _predict_player_performance(self, combined_sequence, position):
    prediction = self.models[position].predict(combined_sequence)
    prediction = np.maximum(prediction, 0)

    target_scaler = self.target_scalers[position]
    prediction_unscaled = target_scaler.inverse_transform(prediction)

    return prediction_unscaled

  def _create_prediction_entry(self, player_id, kickoff_time, prediction, position):
    targets = Features.get_targets_for_position(position)

    return {
      'player_id': player_id,
      'kickoff_time': kickoff_time,
      **{target: pred for target, pred in zip(targets, prediction)}
    }

  def _get_player_sequence(self, player_data, position):
    player_data = player_data.sort_values(by='kickoff_time').tail(self.time_steps)

    features = Features.get_features_for_position(position)
    sequence = player_data[features].values

    return sequence

  def _update_data_with_predictions(self, predictions, current_data, kickoff_time):
    # Update or add player performance data based on predictions
    for prediction in predictions:
      player_id = prediction['player_id']
      player_index = current_data[
        (current_data['id'] == player_id) & 
        (current_data['kickoff_time'] == kickoff_time)
      ].index

      if not player_index.empty:
        # Update relevant stats in the player's row
        for key, value in prediction.items():
          if key not in ['player_id', 'kickoff_time']:
              current_data.loc[player_index, key] = value
      else:
        position = self._get_player_position(player_id)

        # If no current entry, create a new one
        new_row = {col: prediction.get(col, 0) for col in Features.get_features_for_position(position)}
        new_row.update({
          'id': player_id,
          'kickoff_time': kickoff_time, 
          'position': position,
          'season': self.season
        })

        new_row_df = pd.DataFrame([new_row])
        current_data = pd.concat([current_data, new_row_df], ignore_index=True)

    return current_data

  def _initialize_with_last_gws(self):
    sorted_gws = self.gw_data.sort_values(by='kickoff_time', ascending=False)
    last_gws = sorted_gws.groupby('id').head(self.time_steps)

    return last_gws

  def _get_player_position(self, player_id):
    position = self.players_data[self.players_data['id'] == player_id]['position']
    return position.iloc[0]