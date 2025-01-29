import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from utils.feature_selector import FeatureSelector
from utils.feature_scaler import FeatureScaler

class LSTMModel:
  def __init__(self, gw_data, season_data, teams_data, fixtures, players_data, season, time_steps=7, train=False):
    self.gw_data = gw_data.sort_values(by=['id', 'kickoff_time'])  # Gameweek-level data
    self.season_data = season_data
    self.teams_data = teams_data
    self.fixtures = fixtures
    self.players_data = players_data
    self.season = season
    self.feature_selector = FeatureSelector()

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
        targets = self.feature_selector.get_targets_for_position(position)
        combined_columns = list(set(features + targets))

        scaler = self.scalers[position]
        scaler.load_scalers(combined_columns)

  def _build_model(self, position):
    features = self.feature_selector.get_features_for_position(position)
    targets = self.feature_selector.get_targets_for_position(position)

    # Define the input layer
    input_layer = Input(shape=(self.time_steps, len(features)))

    # Shared LSTM layers
    shared_lstm = LSTM(64, return_sequences=True)(input_layer)
    shared_lstm = Dropout(0.5)(shared_lstm)
    shared_lstm = LSTM(32)(shared_lstm)

    # Single output layer for position-specific targets
    output_layer = Dense(len(targets), activation='linear', name=f"{position}_output")(shared_lstm)

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
    targets = self.feature_selector.get_targets_for_position(position)
    combined_columns = list(set(features + targets))

    feature_scaler = self._fit_scaler(position_gw_data, combined_columns, position)

    scaled_data = position_gw_data.copy()
    scaled_data[combined_columns] = feature_scaler.transform(position_gw_data[combined_columns])

    for player_id, player_data in scaled_data.groupby('id'):
      player_data = player_data.reset_index(drop=True)

      # Use all feature columns for input
      X = player_data[features].values
      
      # Use only target columns for prediction targets
      y = player_data[targets].values

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
    merged_data = self.gw_data.merge(self.players_data[['id', 'position']], on='id')
    filtered_data = merged_data[merged_data['position'] == position]
    return filtered_data

  def predict_season(self):
    print(f"Started Season Prediction")

    # Initialize a dictionary to accumulate predictions for each player
    player_aggregates = {}

    current_data = self._initialize_with_last_gws()
    current_data['kickoff_time'] = pd.to_datetime(current_data['kickoff_time'], errors='coerce')

    for _, game in self.fixtures.sort_values(by=['kickoff_time']).iterrows():
      home_team_id = game['team_h']
      away_team_id = game['team_a']
      kickoff_time = pd.to_datetime(game['kickoff_time'])

      home_players = current_data[
        (current_data['team'] == home_team_id) &
        (current_data['kickoff_time'].dt.date <= kickoff_time.date())
      ]
      away_players = current_data[
        (current_data['team'] == away_team_id) & 
        (current_data['kickoff_time'].dt.date <= kickoff_time.date())
      ]

      # Predict and accumulate results for players in both teams
      home_players_predictions = self._predict_players_game(home_players, kickoff_time, player_aggregates)
      away_players_predictions = self._predict_players_game(away_players, kickoff_time, player_aggregates)

      predictions = pd.concat(home_players_predictions + away_players_predictions, ignore_index=True, sort=False)

      self._format_and_save_match_prediction(predictions, home_team_id, away_team_id, kickoff_time)

      current_data = self._update_data_with_predictions(predictions, current_data, kickoff_time, home_team_id, away_team_id)

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

  def _predict_players_game(self, players, kickoff_time, player_aggregates):
    predictions = []

    for player_id, player_data in players.groupby('id'):
      position = self._get_player_position(player_id)
      targets = self.feature_selector.get_targets_for_position(position)

      player_sequence = self._get_player_sequence(player_data, position)
  
      combined_sequence = player_sequence[np.newaxis, :, :]

      # Returns unscaled prediction
      prediction = self._predict_player_performance(combined_sequence, position, targets)

      prediction_entry = self._create_prediction_entry(player_id, kickoff_time, prediction)

      predictions.append(prediction_entry)

      # Add to aggregate
      if player_id not in player_aggregates:
        # Initialize player's aggregate with scalar values (not Series)
        player_aggregates[player_id] = {metric: prediction_entry[metric].values[0] for metric in targets}
      else:
        # Update existing aggregate by adding the scalar values from the Series
        for metric in targets:
          player_aggregates[player_id][metric] += prediction_entry[metric].values[0]

    
    return predictions

  def _predict_player_performance(self, combined_sequence, position, targets):
    prediction = self.models[position].predict(combined_sequence)    
    prediction = np.maximum(prediction, 0) # Ensures that there are no negative numbers

    prediction_df = pd.DataFrame(prediction, columns=targets)

    scaler = self.scalers[position]
    prediction_unscaled = scaler.inverse_transform(prediction_df)

    return prediction_unscaled

  def _create_prediction_entry(self, player_id, kickoff_time, prediction):
    prediction['player_id'] = player_id
    prediction['kickoff_time'] = kickoff_time

    return prediction

  def _get_player_sequence(self, player_data, position):
    player_data = player_data.sort_values(by='kickoff_time').tail(self.time_steps)

    features = self.feature_selector.get_features_for_position(position)

    scaler = self.scalers[position]
    scaled_sequence = scaler.transform(player_data[features])

    return scaled_sequence.values

  def _update_data_with_predictions(self, predictions, current_data, kickoff_time, home_team_id, away_team_id):
    # Iterate over the rows of the predictions DataFrame
    for _, prediction in predictions.iterrows():
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
        # Create a new row with prediction values and additional information
        prediction['id'] = player_id
        prediction['kickoff_time'] = kickoff_time
        prediction['season'] = self.season
        prediction['team'] = self._get_player_team(player_id)
        prediction['value'] = current_data['value'].iloc[-1]

        team_id = self._get_player_team(player_id)
        was_home = 1 if team_id == home_team_id else 0

        prediction['was_home'] = was_home

        new_player_row = pd.DataFrame([prediction])

        # Update player data with previous season data if necessary
        new_player_row = self._add_season_stats(new_player_row)

        # Add team context data to prediction
        new_player_row = self._add_team_contexts(new_player_row, home_team_id, away_team_id)

        # Concatenate the new row to the current data
        current_data = pd.concat([current_data, new_player_row], ignore_index=True)

    return current_data
  
  def _add_team_contexts(self, player_data, home_team_id, away_team_id):
    # Extract home and away team data as Series
    home_team_data = self.teams_data[self.teams_data['id'] == home_team_id].iloc[0]
    away_team_data = self.teams_data[self.teams_data['id'] == away_team_id].iloc[0]

    # Select relevant features and rename the Series index with a prefix
    home_team_context = home_team_data[self.feature_selector.TEAM_FEATURES]
    home_team_context.index = home_team_context.index.map(lambda x: f"home_team_{x}")

    away_team_context = away_team_data[self.feature_selector.TEAM_FEATURES]
    away_team_context.index = away_team_context.index.map(lambda x: f"away_team_{x}")

    # Concatenate the home and away team contexts
    combined_teams_context = pd.concat([home_team_context, away_team_context])

    for col, value in combined_teams_context.items():
      player_data[col] = value
    
    return player_data


  def _initialize_with_last_gws(self):
    sorted_gws = self.gw_data.sort_values(by='kickoff_time', ascending=False)
    last_gws = sorted_gws.groupby('id').head(self.time_steps)

    return last_gws

  def _get_player_position(self, player_id):
    position = self.players_data[self.players_data['id'] == player_id].iloc[0]['position']

    if position == None:
      raise Exception('Position could not be obtained for id: {player_id}')

    return position

  def _get_player_team(self, player_id):
    team = self.players_data[self.players_data['id'] == player_id].iloc[0]['team']

    if team == None:
      raise Exception('Team could not be obtained for id: {player_id}')

    return team

  def _add_season_stats(self, player_data):
    season_data = self.season_data[self.feature_selector.SEASON_FEATURES]
    season_data = self.season_data.rename(columns=lambda x: f"season_{x}")

    player_data = player_data.merge(
      season_data,
      how='left',
      left_on='id',
      right_on='season_id'
    )

    return player_data
  
  def _format_and_save_match_prediction(self, predictions, home_team_id, away_team_id, kickoff_time):
    directory = f"predictions/matches/{self.season}"
    os.makedirs(directory, exist_ok=True)

    home_team = self.teams_data[self.teams_data['id'] == home_team_id].iloc[0]['name']
    away_team = self.teams_data[self.teams_data['id'] == away_team_id].iloc[0]['name']

    file_path = os.path.join(directory, f"{kickoff_time.strftime("%Y-%m-%d_%H:%M:%S_%Z")}_{home_team}_vs_{away_team}.csv")

    predictions = predictions.sort_values(by=['total_points'], ascending=False)
    predictions.to_csv(file_path, index=False)
    print(f"{home_team} vs {away_team} saved to {file_path}\n")

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
