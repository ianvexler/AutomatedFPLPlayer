import io
import numpy as np
import requests
import csv
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/"

class DataLoader():
  def __init__(self, season):  
    self.season = season

  ### FULL SEASON ###
  """
  Gets and formats the data for a full season

  Returns:
    Data frame containing a season's data
  """
  def get_full_season_data(self):
    raw_data = self.fetch_players_raw()
    
    data = self.sanitize_season_data(raw_data)
    return data

  """
  Sanitizes the season data by filtering the relevant columns provided by the DATA_COLUMNS constants.

  Params:
    data_df: DataFrame with the raw data

  Returns:
    pd.DataFrame: Sanitized DataFrame of data.
  """
  def sanitize_season_data(self, data_df):
    # Ensure data_df is in DataFrame format (if not already)
    if not isinstance(data_df, pd.DataFrame):
        data_df = pd.DataFrame(data_df[1:], columns=data_df[0])  # Adjust if raw data isn't already a DataFrame

    # Convert DataFrame values to numeric where possible
    data_df = self.convert_df_to_numeric(data_df)

    # Select only the columns that are in DATA_COLUMNS
    sanitized_df = data_df.loc[:, data_df.columns.intersection(DATA_COLUMNS)]

    return sanitized_df

  ### GWs Merged ###
  def get_merged_gw_data(self):
    raw_data = self.fetch_merged_gw()

    data = self.sanitize_gw_data(raw_data)

    return data

  """
  Sanitizes the gw data by filtering the relevant columns provided by the GW_COLS constants.

  Params:
    data_df: DataFrame with the raw data

  Returns:
    pd.DataFrame: Sanitized DataFrame of data.
  """
  def sanitize_gw_data(self, data_df):
    # Ensure data_df is in DataFrame format (if not already)
    if not isinstance(data_df, pd.DataFrame):
      data_df = pd.DataFrame(data_df[1:], columns=data_df[0])  # Adjust if raw data isn't already a DataFrame

    # Select only the columns that are in GW_COLS
    sanitized_df = data_df.loc[:, data_df.columns.intersection(GW_COLS)]

    # Set 'element' column as the index and sort the DataFrame
    sanitized_df = sanitized_df.rename(columns={'element': 'id'})

    # Convert only the boolean columns to integers
    boolean_columns = sanitized_df.select_dtypes(include='bool').columns
    sanitized_df[boolean_columns] = sanitized_df[boolean_columns].astype(int)

    sanitized_df['kickoff_time'] = pd.to_datetime(sanitized_df['kickoff_time'], format='ISO8601', utc=True)

    sanitized_df['season'] = self.season

    # TODO: Handle setting the id as the indexes?
    # sanitized_df = sanitized_df.set_index('id')  # Remove inplace=True

    return sanitized_df

  ### Teams ###
  def get_teams_data(self):
    raw_data = self.fetch_teams()

    teams_df = self.sanitize_teams_data(raw_data)

    return teams_df

  def sanitize_teams_data(self, data):
    teams_df = pd.DataFrame(data)

    sanitized_data = teams_df.loc[:, teams_df.columns.intersection(TEAMS_COLUMNS)]

    return sanitized_data

  ### ID Dict ###
  def get_id_dict_data(self):
    raw_data = self.fetch_id_dict()
    ids_df = pd.DataFrame(raw_data)

    return ids_df

  ### Fixtures ###
  def get_fixtures_data(self):
    raw_data = self.fetch_fixtures()
    fixtures_df = self.sanitize_fixtures_data(raw_data)

    return fixtures_df

  def sanitize_fixtures_data(self, data):
    fixtures_df = pd.DataFrame(data)

    sanitized_data = fixtures_df.loc[:, fixtures_df.columns.intersection(FIXTURES_COLUMNS)]
    sanitized_data = sanitized_data.rename(columns={'event': 'GW'})

    sanitized_data['kickoff_time'] = pd.to_datetime(sanitized_data['kickoff_time'], format='ISO8601', utc=True)

    sorted_data = sanitized_data.sort_values('kickoff_time', ascending=True)

    return sorted_data
    
  ### Cleaned Players ###
  def get_players_data(self):
    raw_data = self.fetch_players()

    players_df = pd.DataFrame(raw_data)

    # Temp
    POSITIONS = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    players_df = players_df.loc[:, players_df.columns.intersection(['id', 'element_type', 'team'])]
    players_df = players_df.rename(columns={'element_type': 'position'})

    players_df['position'] = players_df['position'].apply(lambda x: POSITIONS[x])
    
    return players_df

  ### FETCH DATA ###
  def fetch_players_raw(self):
    players_raw_url = f"{self.season}/players_raw.csv"
    return self.get_csv_data(players_raw_url)
  
  def fetch_id_dict(self):
    id_dict_url = f"{self.season}/id_dict.csv"
    return self.get_csv_data(id_dict_url)

  def fetch_merged_gw(self):
    merged_gw_url = f"{self.season}/gws/merged_gw.csv"
    return self.get_csv_data(merged_gw_url)

  def fetch_teams(self):
    teams_url = f"{self.season}/teams.csv"
    return self.get_csv_data(teams_url)

  def fetch_fixtures(self):
    fixtures_url = f"{self.season}/fixtures.csv"
    return self.get_csv_data(fixtures_url)

  def fetch_players(self):
    players_url = f"{self.season}/players_raw.csv"
    return self.get_csv_data(players_url)

  ### UTILS ###
  def get_csv_data(self, url):
    response = requests.get(f"{BASE_URL}{url}")
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))

    return df
  
  def convert_df_to_numeric(self, df):
    # Apply pd.to_numeric to each cell of the DataFrame
    df = df.map(lambda x: pd.to_numeric(x, errors='coerce'))
    return df

DATA_COLUMNS = [
  "assists",
  "bonus",
  "bps",
  "clean_sheets",
  "creativity",
  "expected_assists",
  "expected_goal_involvements",
  "expected_goals",
  "expected_goals_conceded",
  "goals_conceded",
  "goals_scored",
  "ict_index",
  "influence",
  "minutes",
  "own_goals",
  "penalties_missed",
  "penalties_saved",
  "red_cards",
  "saves",
  "starts",
  "threat",
  "total_points",
  "yellow_cards",
  # "chance_of_playing_next_round", # TODO?
  # "chance_of_playing_this_round", # TODO?
  "clean_sheets_per_90",
  # "corners_and_indirect_freekicks_order",
  # "cost_change_event",
  # "cost_change_event_fall",
  # "cost_change_start",
  # "cost_change_start_fall",
  # "creativity_rank",
  # "creativity_rank_type",
  # "direct_freekicks_order",
  "element_type",
  "expected_assists_per_90",
  "expected_goal_involvements_per_90",
  "expected_goals_conceded_per_90",
  "expected_goals_per_90",
  "form",
  # "form_rank",
  # "form_rank_type",
  "goals_conceded_per_90",
  # "ict_index_rank",
  # "ict_index_rank_type",
  "id",
  # "influence_rank",
  # "influence_rank_type",
  # "news", # TODO?
  # "news_added", # TODO?
  # "penalties_order",
  "points_per_game",
  "saves_per_90",
  "starts_per_90",
  # "status", # TODO?
  "team_code",
  # "threat_rank",
  # "threat_rank_type",
]

# Maybe include position?
GW_COLS = [
  "team", "opponent_team", "assists", "clean_sheets", 
  "creativity", "element", "expected_assists", "expected_goal_involvements", 
  "expected_goals", "expected_goals_conceded", "goals_conceded", 
  "goals_scored", "ict_index", "influence", "minutes", 
  "own_goals", "penalties_missed", "penalties_saved", 
  "red_cards", "saves", "starts", "threat", "total_points",
  "value", "was_home", "yellow_cards", "kickoff_time"
]

TEAMS_COLUMNS = ['id', 'name', 'strength', 'strength_overall_home', 'strength_overall_away',	'strength_attack_home',	'strength_attack_away',	'strength_defence_home', 'strength_defence_away']

FIXTURES_COLUMNS = ['id', 'event', 'kickoff_time', 'team_a', 'team_h', 'team_h_difficulty',	'team_a_difficulty']