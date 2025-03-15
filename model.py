import os
import argparse
import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from tensorflow.keras.preprocessing.sequence import pad_sequences
from xgboost import XGBRegressor

from utils.model_types import ModelType
from utils.feature_selector import FeatureSelector
from utils.feature_scaler import FeatureScaler
from models.lstm_model import LSTMModel
from data.data_loader import DataLoader

class Model:
  def __init__(
    self, 
    gw_data, 
    teams_data, 
    fixtures, 
    players_data, 
    season,
    time_steps=7, 
    train=False, 
    include_prev_season=False,
    model_type = ModelType.LSTM,
    include_fbref=False,
    training_years=3):
    self.model_type = model_type

    self.gw_data = gw_data.sort_values(by=['id', 'kickoff_time'])  # Gameweek-level data
    self.teams_data = teams_data
    self.fixtures = fixtures
    self.players_data = players_data
    self.season = season
    self.feature_selector = FeatureSelector()

    self.target = self.feature_selector.TARGET

    self.time_steps = time_steps
    self.include_prev_season = include_prev_season
    self.include_fbref = include_fbref
    self.training_years = training_years

    self.models = {
      'GK': self._set_model('GK'),
      'DEF': self._set_model('DEF'),
      'MID': self._set_model('MID'),
      'FWD': self._set_model('FWD')
    }

    self.scalers = {
      'GK': FeatureScaler('GK', self.model_type),
      'DEF': FeatureScaler('DEF', self.model_type),
      'MID': FeatureScaler('MID', self.model_type),
      'FWD': FeatureScaler('FWD', self.model_type)
    }

    self.DIRECTORY = f"{self.model_type.value}/steps_{self.time_steps}_prev_season_{self.include_prev_season}_fbref_{self.include_fbref}"
    
    if not train:
      for position in self.models:
        directory = f"models/trained/{self.DIRECTORY}"

        self.models[position] = self._load_model(position)
        
        features = self.feature_selector.get_features_for_position(position, self.include_prev_season, self.include_fbref)      
        scaler = self.scalers[position]
        scaler.load_scalers(features)
  
  def _set_model(self, position):
    match self.model_type.value:
      case ModelType.RANDOM_FOREST.value:
        return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
      case ModelType.ADABOOST.value:
        return AdaBoostRegressor(n_estimators=50, learning_rate=0.01)
      case ModelType.GRADIENT_BOOST.value:
        return GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
      case ModelType.XGBOOST.value:
            return XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
      case ModelType.LSTM.value:
        return LSTMModel(self.time_steps, position, self.include_prev_season).build_model()
      case ModelType.ML_LSTM.value:
        return LSTMModel(self.time_steps, position, self.include_prev_season, multi_layer=True).build_model()
    raise Exception(f'No model matches {self.model_type}')

  def train(self):
    training_data = self._get_training_data()

    for position in ['GK', 'DEF', 'MID', 'FWD']:
      print(f"Training model and fitting scalers for {position}\n")

      # Train the model
      model = self.models[position]
      
      X, y = self._prepare_sequences(training_data, position)

      if self.model_type == ModelType.LSTM.value:
        model.fit(X, y, epochs=80, batch_size=32, validation_split=0.2, verbose=1)
      else:
        model.fit(X, y)

      directory = f"models/trained/{self.DIRECTORY}"
      os.makedirs(directory, exist_ok=True)

      self._save_model(position)
      print(f"  Saved model and scalers for {position}\n")

  def _save_model(self, position):
    directory = f"models/{self.DIRECTORY}"
    os.makedirs(directory, exist_ok=True)  # Ensure directory exists

    model_path = os.path.join(directory, f'{self.model_type.value}_{position}_steps_{self.time_steps}')

    if self._is_model_sequential():
      # Save LSTM model
      self.models[position].save(model_path + ".keras")
    else:
      # Save Scikit-Learn model
      joblib.dump(self.models[position], model_path + ".joblib")

  def _load_model(self, position):
    directory = f"models/{self.DIRECTORY}"
    model_path = os.path.join(directory, f'{self.model_type.value}_{position}_steps_{self.time_steps}')

    if self._is_model_sequential():
      # Load LSTM model
      self.models[position] = tf.keras.models.load_model(model_path + ".keras")
    else:
      # Load Scikit-Learn model
      self.models[position] = joblib.load(model_path + ".joblib")

  def _get_training_data(self):
    start_year, end_year = self.season.split('-')
    start_year, end_year = int(start_year), int(end_year)

    past_start_year = start_year - self.training_years

    data_loader = DataLoader()
    all_data = []
    required_columns = None

    # Train using data from n years in the past
    while start_year >= past_start_year:
      prev_season = f"{start_year - 1}-{end_year - 1}"
      print(f"Loading training data for season: {prev_season}")

      # TODO: Add option to not include team data
      season_data = data_loader.get_merged_gw_data(
        prev_season, 
        time_steps=self.time_steps, 
        include_season=self.include_prev_season,
        include_teams=True,
        include_fbref=self.include_fbref)

      # Ensure the data has the required columns
      if all_data:
        required_columns = all_data[0].columns
        missing_columns = set(required_columns) - set(season_data.columns)

        if missing_columns:
          print(f"Missing columns in {prev_season}: {missing_columns}")
          for col in missing_columns:
            season_data[col] = 0

      all_data.append(season_data)

      # Move to the previous season
      start_year -= 1
      end_year -= 1

    return all_data 

    # Concatenate all collected data and return
    if all_data:
      training_data = pd.concat(all_data, ignore_index=True)
      return training_data
    else:
      raise ValueError("No valid training data could be loaded.\n")

  def _prepare_sequences(self, training_data, position):
    X = []
    y = []

    # Filter data per position
    position_gw_data = training_data[training_data['position'] == position]
    
    features = self.feature_selector.get_features_for_position(position, self.include_prev_season, self.include_fbref)
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
        
        # Flatten if not LSTM
        if not self._is_model_sequential():
          X_sequence = X_sequence.flatten()
   
        X.append(X_sequence)
        y.append(scaled_target)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

  def _fit_scaler(self, position_gw_data, features, position):
    scaler = self.scalers[position]
    scaler.fit(position_gw_data, features)
    return scaler

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
      
      if self._is_model_sequential():
        # LSTM needs 3D input
        X_input = padded_sequences
      else:
        X_input = padded_sequences.reshape(padded_sequences.shape[0], -1) 

      # Perform batch prediction
      predictions_batch = self._predict_player_performance(X_input, position)

      # Assign predictions back to respective players
      for i, player_id in enumerate(player_ids):
        predictions.setdefault(player_id, 0)
        predictions[player_id] += predictions_batch[i]

    return predictions

  def _predict_player_performance(self, data, position):
    predictions = self.models[position].predict(data)
    # predictions = np.maximum(predictions, 0)  # Avoid negative values

    prediction_df = pd.DataFrame(predictions, columns=[self.target])

    scaler = self.scalers[position]
    predictions_unscaled = scaler.inverse_transform(prediction_df, 'target')
    
    return np.array(predictions_unscaled).flatten()

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

    features = self.feature_selector.get_features_for_position(position, self.include_prev_season, self.include_fbref)
    player_data = player_data[features]

    scaler = self.scalers[position]
    scaled_sequence = scaler.transform_data(player_data)
    return scaled_sequence.values

  def _get_player_position(self, player_id):
    position = self.players_data[self.players_data['id'] == player_id].iloc[0]['position']

    if position == None:
      raise Exception('Position could not be obtained for id: {player_id}')

    return position
  
  def _is_model_sequential(self):
    return self.model_type in { ModelType.LSTM, ModelType.ML_LSTM }
  
  def _format_and_save_match_prediction(self, predictions, home_team_id, away_team_id, gw_data, current_gw):
    directory = f"predictions/{self.DIRECTORY}/matches/{self.season}"
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
    directory = f"predictions/{self.DIRECTORY}/gws/{self.season}"
    os.makedirs(directory, exist_ok=True)

    file_path = os.path.join(directory, f"GW{int(current_gw)}.csv")
    predictions_df = pd.DataFrame.from_dict(predictions, orient='index', columns=['expected_points'])

    gw_predictions_df = gw_data.merge(predictions_df, left_on='id', right_index=True, how='left')
    gw_predictions_df.to_csv(file_path, index=False)
    
    print(f"GW{int(current_gw)} saved to {file_path}\n")
    
    return predictions

  def _format_and_save_predictions(self, aggregate_predictions):
    directory = f'predictions/{self.DIRECTORY}/'
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"predictions_{self.season}.csv")

    # Convert aggregate predictions to a DataFrame
    predictions_df = pd.DataFrame(aggregate_predictions)
    predictions_df.to_csv(file_path, index=False)
    print(f"Predictions saved to {file_path}")

    return predictions_df

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Run the model with optional training.')
  parser.add_argument('--steps', type=int, nargs='?', const=5, default=5, help='Time step for data window. Defaults to 7 if not provided or null.')
  parser.add_argument('--season', type=str, nargs='?', default='2024-25', help='Season to simulate in the format 20xx-yy')
  parser.add_argument('--prev_season', action='store_true', help='Set this flag to include prev season data. Defaults to false.')
  parser.add_argument('--model', type=str, help='The model to use', choices=[m.value for m in ModelType])
  parser.add_argument('--no_train', action='store_true', help='')
  parser.add_argument('--fbref', action='store_true', help='')
  
  args = parser.parse_args()

  try:
    model_type = ModelType(args.model)
  except ValueError:
    print(f"Error: Invalid model type '{args.model}'. Choose from {', '.join(m.value for m in ModelType)}")
    exit(1)

  data_loader = DataLoader()

  fixtures_data = data_loader.get_fixtures(args.season)
  teams_data = data_loader.get_teams_data(args.season)
  players_data = data_loader.get_players_data(args.season)
  gw_data = data_loader.get_merged_gw_data(args.season, args.steps, include_season=args.prev_season, include_fbref=args.fbref)

  model = Model(
    gw_data=gw_data,
    teams_data=teams_data,
    fixtures=fixtures_data,
    players_data=players_data,
    season=args.season,
    time_steps=args.steps,
    include_prev_season=args.prev_season,
    model_type=model_type,
    train=(not args.no_train),
    include_fbref=args.fbref)

  if not args.no_train:
    model.train()
  model.predict_season()