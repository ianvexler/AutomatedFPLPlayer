import os
import argparse
import joblib
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from xgboost import XGBRegressor

from utils.model_types import ModelType
from utils.feature_selector import FeatureSelector
from utils.feature_scaler import FeatureScaler
from models.lstm_model import LSTMModel
from data.data_loader import DataLoader

class Model:
  def __init__(
    self, 
    season,
    time_steps=7, 
    include_prev_season=False,
    model_type = ModelType.LSTM,
    include_fbref=False,
    training_years=3,
    include_season_aggs=False,
    include_teams=False,
    gw_lamda_decay=0.02,
    top_features=None,
    no_cache=False,
    lstm_config=None
  ):
    self.model_type = model_type

    self.season = season
    self.feature_selector = FeatureSelector()

    self.target = self.feature_selector.TARGET

    self.time_steps = time_steps
    self.include_prev_season = include_prev_season
    self.include_fbref = include_fbref
    self.include_season_aggs = include_season_aggs
    self.include_teams = include_teams
    self.gw_lamda_decay = gw_lamda_decay
    self.top_features = top_features
    
    self.training_years = training_years
    self.no_cache = no_cache

    self.lstm_config = lstm_config

    self.models = {
      'GK': self._set_model('GK'),
      'DEF': self._set_model('DEF'),
      'MID': self._set_model('MID'),
      'FWD': self._set_model('FWD')
    }

    if self._is_model_sequential():
      self.scalers = {
        'GK': FeatureScaler('GK'),
        'DEF': FeatureScaler('DEF'),
        'MID': FeatureScaler('MID'),
        'FWD': FeatureScaler('FWD')
      }

    gw_decay_str = str(self.gw_lamda_decay).replace('.', '_')
    self.FILE_NAME = f"steps_{self.time_steps}_prev_season_{self.include_prev_season}_fbref_{self.include_fbref}_season_aggs_{self.include_season_aggs}_teams_{self.include_teams}_gw_decay_{gw_decay_str}_top_features_{self.top_features}"
    self.DIRECTORY = f"{self.model_type.value}/{self.FILE_NAME}"
      
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

    if self._is_model_sequential():
      return LSTMModel(
          time_steps=self.time_steps, 
          position=position,
          features=self._get_position_features(position),
          model_type=self.model_type,
          config=self.lstm_config
        ).build_model()
      
    raise Exception(f'No model matches {self.model_type}')

  def _load_data(self):
    data_loader = DataLoader(no_cache=self.no_cache)

    fixtures_data = data_loader.get_fixtures(self.season)
    teams_data = data_loader.get_teams_data(self.season)
    players_data = data_loader.get_players_data(self.season)
    gw_data = data_loader.get_merged_gw_data(
      self.season, 
      self.time_steps, 
      include_prev_season=self.include_prev_season, 
      include_fbref=self.include_fbref,
      include_season_aggs=self.include_season_aggs,
      include_teams=self.include_teams,
      include_prev_gws=True
    )

    self.gw_data = gw_data.sort_values(by=['id', 'kickoff_time'])  # Gameweek-level data
    self.teams_data = teams_data
    self.fixtures = fixtures_data
    self.players_data = players_data

  def train(self):
    """
    Trains the model by getting all player sequences from the last n season
    
    Returns
      Histories to be used in evaluation
    """
    training_data = self._get_training_data()
  
    pos_training_data = {}
    pos_model_histories = {}

    for position in ['GK', 'DEF', 'MID', 'FWD']:
      print(f"Training model and fitting scalers for {position}\n")

      # Train the model
      model = self.models[position]
      X, y = self._prepare_training_sequences(training_data, position)
      
      if np.isnan(X).sum() > 0 or np.isnan(y).sum() > 0:
        raise ValueError(f"Found NaN values in training data for {position}")
      
      if self._is_model_sequential():
        # Apply callbacks 
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

        print(f"Total number of training samples: {len(y)}")
        history = model.fit(X, y, 
          validation_split=0.2, 
          epochs=50, 
          batch_size=128,
          callbacks=[early_stopping, lr_schedule])
      else:
        history = model.fit(X, y)

      pos_model_histories[position] = history 
      pos_training_data[position] = (X, y)

    return pos_training_data, pos_model_histories

  def _get_training_data(self):
    start_year, end_year = self.season.split('-')
    start_year, end_year = int(start_year), int(end_year)

    past_start_year = start_year - self.training_years

    data_loader = DataLoader(self.no_cache)
    all_data = []
    required_columns = None

    total_samples = 0

    # Train using data from n years in the past
    while start_year > past_start_year:
      prev_season = f"{start_year - 1}-{end_year - 1}"
      print(f"Loading training data for season: {prev_season}")

      # Do not include prev gws from COVID season
      include_prev_gws = True if (start_year - 2) != 19 else False

      season_data = data_loader.get_merged_gw_data(
        prev_season, 
        time_steps=self.time_steps, 
        include_prev_season=self.include_prev_season,
        include_fbref=self.include_fbref,
        include_prev_gws=include_prev_gws,
        include_season_aggs=self.include_season_aggs,
        include_teams=self.include_teams
      )

      # Ensure the data has the required columns
      if all_data:
        required_columns = all_data[0].columns
        missing_columns = set(required_columns) - set(season_data.columns)

        # Add missing columns if necessary
        if missing_columns:
          season_data = season_data.reindex(columns=required_columns, fill_value=0)

      if not include_prev_gws:
        season_data = season_data[season_data['GW'] > self.time_steps]

      all_data.append(season_data)
      total_samples += len(season_data)

      # Move to the previous season
      start_year -= 1
      end_year -= 1

    # Concatenate all collected data and return
    if all_data:
      training_data = pd.concat(all_data, ignore_index=True)
      return training_data.drop_duplicates()
    else:
      raise ValueError("No valid training data could be loaded.\n")

  def _get_position_features(self, position):
    """
    Loads features from feature_selector based on configuration
    """
    return self.feature_selector.get_features_for_position(
      position, 
      include_prev_season=self.include_prev_season, 
      include_fbref=self.include_fbref, 
      include_season_aggs=self.include_season_aggs,
      include_teams=self.include_teams,
      top_n=self.top_features)

  def _prepare_training_sequences(self, training_data, position):
    """
    Gets training player sequences
    """
    cache_dir = "data/cached"
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_filename = os.path.join(cache_dir, f"{self.season}_{position}_{self.FILE_NAME}.pkl")

    # Check if sequences have already being generated and cached
    if os.path.exists(cache_filename) and (not self.no_cache):
      print(f"Loading cached sequences for {position} from {cache_filename}")
      with open(cache_filename, "rb") as f:
        data = pickle.load(f)
        X, y = data["X"], data["y"]

      # Fit scaler if model is sequential
      if self._is_model_sequential():
        position_gw_data = training_data[training_data['position'] == position]
        features = self._get_position_features(position)
        self._fit_scaler(position_gw_data, features, position)
    else:
      print(f"Generating sequences for {position}")
      X, y = [], []

      position_gw_data = training_data[training_data['position'] == position]
      features = self._get_position_features(position)

      # Fit scaler if model is sequential
      if self._is_model_sequential():
        self._fit_scaler(position_gw_data, features, position)

      # Iterate through all player data
      for player_id, player_data in position_gw_data.groupby('id'):
        season_kickoffs = player_data[player_data['GW'] >= 1]['kickoff_time'].tolist()

        # Iterate through every game for each player
        for kickoff_time in season_kickoffs:
          sequence = self._get_player_sequence(player_data, position, kickoff_time)

          if sequence.shape[0] < self.time_steps:
            continue

          if np.isnan(sequence).sum() > 0:
            raise ValueError(f"NaN detected for {player_id} at {kickoff_time}")

          target_row = player_data[player_data['kickoff_time'] == kickoff_time].iloc[0]
          y_target = target_row[self.target]

          # If model sequential scale date
          if self._is_model_sequential():
            y_target = self.scalers[position].transform(y_target, 'total_points', target=True)

          X.append(sequence)
          y.append(y_target)

      X = np.array(X, dtype=np.float32)
      y = np.array(y, dtype=np.float32)

      # Save sequences for future use
      with open(cache_filename, "wb") as f:
        pickle.dump({"X": X, "y": y}, f)
        print(f"Saved sequence cache for {position} to {cache_filename}")

    # Flatten if non-sequential
    if not self._is_model_sequential():
      X = X.reshape((X.shape[0], -1))

    return X, y

  def _fit_scaler(self, position_gw_data, features, position):
    scaler = self.scalers[position]
    scaler.fit(position_gw_data, features)
    return scaler

  def _filter_data_by_position(self, position):
    return self.gw_data[self.gw_data['position'] == position]

  def predict_season(self):
    """
    Runs predictions for the entire season across all gameweeks.

    For each gameweek:
    - Loads and filters the relevant data up to that gameweek.
    - Predicts individual player performances for each fixture.
    """
    print(f"Started Season Prediction")

    self._load_data()

    # Initialize a dictionary to accumulate predictions for each player
    player_aggregates = {}

    # Group games by fixtures and iterate through them
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

        # Get data previous to game being predicted
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
    """
    Generates expected points for all players accross all 
    positions in a fixture
    """
    predictions = {}

    position_groups = players_data.groupby('position')
    
    # Iterates through all positions
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

        if not self._is_model_sequential():
          sequence = sequence.flatten()

        player_sequences.append(sequence)
      
      if self._is_model_sequential():
        # LSTM needs 3D input
        X_input = np.array(player_sequences, dtype=np.float32)
      else:
        X_input = np.array([seq.flatten() for seq in player_sequences])

      # Perform batch prediction
      predictions_batch = self._predict_performances(X_input, position)

      # Assign predictions back to respective players
      for i, player_id in enumerate(player_ids):
        predictions.setdefault(player_id, 0)
        predictions[player_id] += predictions_batch[i]

    return predictions

  def _predict_performances(self, data, position):
    """
    Generates expected points for all players in a fixture 
    for specific position
    """
    predictions = self.models[position].predict(data)

    prediction_df = pd.DataFrame(predictions, columns=[self.target])

    if self._is_model_sequential():
      scaler = self.scalers[position]
      predictions = scaler.inverse_transform(prediction_df, 'target')

    return np.round(np.array(predictions).flatten(), 2)

  def _add_gw_decay(self, player_data, kickoff_time):
    """
    Value applied to each sequence that assigns a significance
    to each gameweek depending on how much time has passed

    Based on the concept of half-life
    """
    lambda_decay = self.gw_lamda_decay

    player_data = player_data.copy()

    days_since_gw = (kickoff_time - player_data["kickoff_time"]).dt.days.to_numpy()
    player_data["gw_decay"] = np.exp(-lambda_decay * days_since_gw)

    return player_data

  def _get_player_sequence(self, player_data, position, kickoff_time):
    """
    Generates sequences for a player.

    Given its data and a specific kickoff_time, it returns a 2d list
    of all n previous games for a player
    """
    player_data = self._add_gw_decay(player_data, kickoff_time)
    all_player_data = player_data.copy()

    # Ensure data is sorted by kickoff time
    player_data = player_data.sort_values(by='kickoff_time')
    
    # Only extracts data from previous matches
    player_data = player_data[player_data['kickoff_time'] < kickoff_time].tail(self.time_steps)

    # Get list of features for this position
    features = self._get_position_features(position)
    n_features = len(features)  # Total number of features per timestep

    # If no past data available, return a zero-filled array
    if player_data.empty:
        return np.zeros((self.time_steps, n_features), dtype=np.float32)

    # Extract feature columns
    player_data = player_data[features]
    player_means = all_player_data[features].mean()  # Get average values for missing data

    # Fill NaN values with mean (fallback to 0 if needed)
    player_data.fillna(player_means, inplace=True)
    player_data.fillna(0, inplace=True)

    # Scale features
    if self._is_model_sequential():
      scaler = self.scalers[position]
      sequence = scaler.transform_data(player_data).values  # Convert to NumPy array
    else:
      sequence = player_data.values

    # Ensure the sequence is exactly (time_steps, n_features)
    current_length = sequence.shape[0]

    if current_length < self.time_steps:
      # Pad with zeros at the beginning (pre-padding) to match the required length
      padding = np.zeros((self.time_steps - current_length, n_features), dtype=np.float32)
      
      # Stack the padding at the start
      sequence = np.vstack((padding, sequence))  

    # Always (time_steps, n_features)
    return sequence

  def _get_player_position(self, player_id):
    position = self.players_data[self.players_data['id'] == player_id].iloc[0]['position']

    if position == None:
      raise Exception('Position could not be obtained for id: {player_id}')

    return position
  
  def _is_model_sequential(self):
    return self.model_type in { ModelType.LSTM, ModelType.ML_LSTM, ModelType.BI_LSTM }
  
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
  parser.add_argument('--season', type=str, nargs='?', default='2023-24', help='Season to simulate in the format 20xx-yy.')
  parser.add_argument('--prev_season', action='store_true', help='Set this flag to include prev season data. Defaults to false.')
  parser.add_argument('--model', type=str, help='The model to use', choices=[m.value for m in ModelType])
  parser.add_argument('--fbref', action='store_true', help='Include FBref data.')
  parser.add_argument('--season_aggs', action='store_true', help='Include season aggregate data.')
  parser.add_argument('--teams', action='store_true', help='Include teams data.')
  parser.add_argument('--gw_decay', type=float, default=0.02, help='The lambda decay applied in gw decay')
  parser.add_argument('--no_cache', action='store_true', help="Don't use cached Data Loader data")
  parser.add_argument('--top_features', type=int, nargs='?', default=None, help='Time step for data window. Defaults to 7 if not provided or null.')

  args = parser.parse_args()

  try:
    model_type = ModelType(args.model)
  except ValueError:
    print(f"Error: Invalid model type '{args.model}'. Choose from {', '.join(m.value for m in ModelType)}")
    exit(1)

  model = Model(
    season=args.season,
    time_steps=args.steps,
    include_prev_season=args.prev_season,
    model_type=model_type,
    include_fbref=args.fbref,
    include_season_aggs=args.season_aggs,
    include_teams=args.teams,
    gw_lamda_decay=args.gw_decay,
    top_features=args.top_features,
    no_cache=args.no_cache)

  model.train()
  model.predict_season()