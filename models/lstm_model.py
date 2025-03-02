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

    if not train:
      for position in self.models:
        directory = 'models/trained'

        self.models[position] = tf.keras.models.load_model(os.path.join(directory, f'lstm_model_{position}_steps_{self.time_steps}.keras'))
        
        # Improve
        features = self.feature_selector.get_features_for_position(position)
        
        scaler = self.scalers[position]
        scaler.load_scalers(features)

    self.DIRECTORY = f"predictions/steps {self.time_steps}"

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

  def train(self, training_data):
    for position in ['GK', 'DEF', 'MID', 'FWD']:
      print(f"Training model and fitting scalers for {position}\n")

      # Prepare sequences from historical gameweek data
      X, y = self._prepare_sequences(training_data, position)

      # Train the model
      model = self.models[position]

      history = model.fit(X, y, epochs=80, batch_size=32, validation_split=0.2, verbose=1)

      directory = 'models/trained'
      os.makedirs(directory, exist_ok=True)

      model.save(os.path.join(directory, f'lstm_model_{position}_steps_{self.time_steps}.keras'))

      print(f"Saved model and scalers for {position}\n")

  def _prepare_sequences(self, training_data, position):
    X = []
    y = []

    # Filter data per position
    position_gw_data = training_data[training_data['position'] == position]
    
    features = self.feature_selector.get_features_for_position(position)
    feature_scaler = self._fit_scaler(position_gw_data, features, position)

    # Iterate on every game for every player
    for player_id, player_data in position_gw_data.groupby('id'):
      season_kickoffs = (player_data[player_data['GW'] >= 1])['kickoff_time'].tolist()

      for kickoff_time in season_kickoffs:
        # Use all feature columns for input
        X_sequence = self._get_player_sequence(player_data, position, kickoff_time)
        
        if X_sequence.shape[0] < self.time_steps:
          continue

        # Get the target of current sequence
        target_game_data = player_data[player_data['kickoff_time'] == kickoff_time].iloc[0]
        y_target = target_game_data[self.target]
        scaled_target = self.scalers[position].transform(y_target, 'total_points', target=True)
        
        X.append(X_sequence)
        y.append(scaled_target)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

  def _fit_scaler(self, position_gw_data, features, position):
    scaler = self.scalers[position]
    scaler.fit(position_gw_data, features)
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
        kickoff_time = game['kickoff_time']

        home_team_id = game['team_h']
        away_team_id = game['team_a']

        # Data of players that are going to perform
        players_gw_data = current_data[
          ((current_data['team'] == home_team_id) | (current_data['team'] == away_team_id)) & 
          (current_data['kickoff_time'] == kickoff_time)
        ]

        # Those player's player perfomances
        players_data = current_data[
          current_data['id'].isin(players_gw_data['id'].unique()) &
          (current_data['kickoff_time'] < kickoff_time)
        ]

        # Predict and accumulate results for players in both teams
        predictions = self._predict_players_game(players_data, kickoff_time)

        self._format_and_save_match_prediction(predictions, home_team_id, away_team_id, players_gw_data, current_gw)

        # Add to aggregate
        for player_id, prediction in predictions.items():
          player_data = players_gw_data[(players_gw_data['id'] == player_id) & (players_gw_data['kickoff_time'] == kickoff_time)].iloc[0]

          player_aggregates.setdefault(player_id, {'expected_points': 0, 'xP': 0})
          player_aggregates[player_id]['expected_points'] += prediction
          player_aggregates[player_id]['xP'] += player_data['xP']

          gw_aggregates.setdefault(player_id, 0)
          gw_aggregates[player_id] += prediction

      players_data = current_data[current_data['GW'] == current_gw]
      self._format_and_save_gw_prediction(gw_aggregates, players_data, current_gw)

    # Convert the accumulated results to a DataFrame
    aggregate_predictions = []
    for player_id, prediction in player_aggregates.items():
      aggregate_predictions.append({
        'id': player_id,
        'expected_points': prediction['expected_points'],
        'xP': prediction['xP']
      })

    # Convert to DataFrame and save as CSV
    predictions_df = self._format_and_save_predictions(aggregate_predictions)

    return predictions_df

  def _predict_players_game(self, players_data, kickoff_time):
    predictions = {}

    position_groups = players_data.groupby('position')

    for position, position_data in position_groups:
      player_ids = position_data['id'].unique()
      player_sequences = []

      for player_id in player_ids:
        player_data = position_data[position_data['id'] == player_id]

        if player_data.empty:
          # If no prior data, fill with zeros of correct shape
          sequence_length = self.models[position].input_shape[1]  # Get model expected sequence length
          feature_count = self.models[position].input_shape[2]  # Get feature count
          sequence = np.zeros((sequence_length, feature_count), dtype='float32')
        else:
          sequence = self._get_player_sequence(player_data, position, kickoff_time)

        player_sequences.append(sequence)

      # Ensure all sequences have the same length by padding
      padded_sequences = pad_sequences(player_sequences, padding='post', dtype='float32')

      # Perform batch prediction
      predictions_batch = self._predict_player_performance(padded_sequences, position)

      # Assign predictions back to respective players
      for i, player_id in enumerate(player_ids):
        predictions.setdefault(player_id, 0)
        predictions[player_id] += predictions_batch[i]

    return predictions

  def _predict_player_performance(self, combined_sequences, position):
    predictions = self.models[position].predict(combined_sequences)
    # predictions = np.maximum(predictions, 0)  # Avoid negative values

    prediction_df = pd.DataFrame(predictions, columns=[self.target])

    scaler = self.scalers[position]
    predictions_unscaled = scaler.inverse_transform(prediction_df, 'target')
    
    return np.array(predictions_unscaled).flatten()

  # TODO: Move to shared model
  # To be applied to every sequence
  def _add_gw_decay(self, player_data, kickoff_time):
    # Calculate the difference in days
    player_data["days_since_gw"] = (kickoff_time - player_data["kickoff_time"]).dt.days

    # Apply exponential decay
    lambda_decay = 0.02 
    player_data["gw_decay"] = np.exp(-lambda_decay * player_data["days_since_gw"])
    
    # Drop raw time difference
    player_data = player_data.drop(columns=["days_since_gw"])
    return player_data

  def _get_player_sequence(self, player_data, position, kickoff_time):
    # Ensure data is sorted by kickoff time
    player_data = player_data.sort_values(by='kickoff_time')
    player_data = player_data[player_data['kickoff_time'] < kickoff_time].tail(self.time_steps)
    
    if player_data.empty:
      return np.array([])

    player_data = self._add_gw_decay(player_data, kickoff_time)

    features = self.feature_selector.get_features_for_position(position)

    scaler = self.scalers[position]

    scaled_sequence = scaler.transform_data(player_data[features])
    return scaled_sequence.values

  def _get_player_position(self, player_id):
    position = self.players_data[self.players_data['id'] == player_id].iloc[0]['position']

    if position == None:
      raise Exception('Position could not be obtained for id: {player_id}')

    return position
  
  def _format_and_save_match_prediction(self, predictions, home_team_id, away_team_id, gw_data, current_gw):
    directory = f"{self.DIRECTORY}/matches/{self.season}"
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
    directory = f"{self.DIRECTORY}/gws/{self.season}"
    os.makedirs(directory, exist_ok=True)

    file_path = os.path.join(directory, f"GW{int(current_gw)}.csv")
    predictions_df = pd.DataFrame.from_dict(predictions, orient='index', columns=['expected_points'])

    gw_predictions_df = gw_data.merge(predictions_df, left_on='id', right_index=True, how='left')
    gw_predictions_df.to_csv(file_path, index=False)
    
    print(f"GW{int(current_gw)} saved to {file_path}\n")
    
    return predictions

  def _format_and_save_predictions(self, aggregate_predictions):
    directory = f'{self.DIRECTORY}/'
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"predictions_{self.season}.csv")

    # Convert aggregate predictions to a DataFrame
    predictions_df = pd.DataFrame(aggregate_predictions)
    predictions_df.to_csv(file_path, index=False)
    print(f"Predictions saved to {file_path}")

    return predictions_df
