import io
import numpy as np
import requests
import csv
import pandas as pd
from utils.team_matcher import TeamMatcher

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/"

class DataLoader():
  def __init__(self, season):  
    self.season = season
    self.POSITIONS = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}

    self.team_matcher = TeamMatcher()

  ### FULL SEASON ###

  def get_full_season_data(self):
    """
    Gets and formats the data for a full season
    """
    raw_data = self.fetch_players_raw()
    
    data = self.sanitize_season_data(raw_data)
    return data

  def sanitize_season_data(self, data_df):
    # Ensure data_df is in DataFrame format (if not already)
    if not isinstance(data_df, pd.DataFrame):
      data_df = pd.DataFrame(data_df[1:], columns=data_df[0])  # Adjust if raw data isn't already a DataFrame

    # Convert DataFrame values to numeric where possible
    sanitized_df = self.convert_df_to_numeric(data_df)
    return sanitized_df

  ### GWs Merged ###
  def get_merged_gw_data(self):
    raw_data = self.fetch_merged_gw()
    gw_data = self.sanitize_gw_data(raw_data)

    if 'team' not in gw_data.columns:
      players_raw = self.fetch_players_raw()
      players_raw['id'] = players_raw['id'].astype(int)
      players_raw['team'] = players_raw['team'].astype(int)
      
      missing_teams = players_raw[['id', 'team']]

      gw_data = gw_data.merge(
        missing_teams,
        how='left',
        left_on='id',
        right_on='id'
      )
    else:
      gw_data['team'] = gw_data['team'].apply(lambda x: self.team_matcher.get_fpl_team(x, self.season, key_type='name')["FPL"][self.season]['id'])

    if 'position' not in gw_data.columns:
      players_raw = self.fetch_players_raw()
      players_raw['id'] = players_raw['id'].astype(int)
      players_raw['element_type'] = players_raw['element_type'].astype(int)
      
      missing_positions = players_raw[['id', 'element_type']]
      missing_positions = missing_positions.rename(columns={'element_type': 'position'})
      
      gw_data = gw_data.merge(
        missing_positions,
        how='left',
        left_on='id',
        right_on='id'
      )
      gw_data['position'] = gw_data['position'].apply(lambda x: self.POSITIONS[x])
      
    # To take into account for the COVID disruptions
    if self.season == '2019-20':
      gw_data = gw_data[gw_data['GW'] <= 29]
  
    return gw_data

  def sanitize_gw_data(self, data_df):
    """
    Sanitizes the gw data by filtering the relevant columns provided by the GW_COLS constants.
    """
    # Ensure data_df is in DataFrame format (if not already)
    if not isinstance(data_df, pd.DataFrame):
      data_df = pd.DataFrame(data_df[1:], columns=data_df[0]) # Adjust if raw data isn't already a DataFrame
    
    # Select only the columns that are in GW_COLS
    sanitized_df = data_df.rename(columns={'element': 'id', 'value': 'cost'})

    # Convert only the boolean columns to integers
    boolean_columns = sanitized_df.select_dtypes(include='bool').columns
    sanitized_df[boolean_columns] = sanitized_df[boolean_columns].astype(int)

    sanitized_df['kickoff_time'] = pd.to_datetime(sanitized_df['kickoff_time'], format='ISO8601', utc=True)

    sanitized_df['season'] = self.season

    # Ensure GKs are not named GKPs
    if 'position' in sanitized_df.columns:
      sanitized_df.loc[sanitized_df['position'] == 'GKP', 'position'] = 'GK'

    return sanitized_df

  ### Teams ###
  def get_teams_data(self):
    raw_data = self.fetch_teams()

    teams_df = self.sanitize_teams_data(raw_data)

    return teams_df

  def sanitize_teams_data(self, data):
    teams_df = pd.DataFrame(data)
    return teams_df

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

    sanitized_data = fixtures_df.rename(columns={'event': 'GW'})
    sanitized_data['kickoff_time'] = pd.to_datetime(sanitized_data['kickoff_time'], format='ISO8601', utc=True)

    sorted_data = sanitized_data.sort_values('kickoff_time', ascending=True)

    return sorted_data
    
  ### Cleaned Players ###
  def get_players_data(self):
    raw_data = self.fetch_players()

    players_df = pd.DataFrame(raw_data)
    
    # TODO: handle managers
    players_df = players_df[players_df['element_type'] <= 4]
    players_df = players_df.rename(columns={'element_type': 'position', 'now_cost': 'cost'})

    players_df['position'] = players_df['position'].apply(lambda x: self.POSITIONS[x])
    
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

    df = pd.read_csv(io.StringIO(response.text), on_bad_lines="skip")
    return df
  
  def convert_df_to_numeric(self, df):
    # Apply pd.to_numeric to each cell of the DataFrame
    df = df.map(lambda x: pd.to_numeric(x, errors='coerce'))
    return df
