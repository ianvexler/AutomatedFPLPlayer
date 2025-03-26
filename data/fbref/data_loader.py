import soccerdata as sd
import os
import pandas as pd
import re
import numpy as np
import warnings
from utils.team_matcher import TeamMatcher
import requests
import time
import unicodedata

class DataLoader:
  def __init__(self, season, no_cache=False, timeout=10):
    self.season = season
    self.fbref = sd.FBref(leagues=[
      'Big 5 European Leagues Combined',
    ], seasons=season)
    self.team_matcher = TeamMatcher()

    self.LEAGUES = {
      'Premier League': 9,
      'Serie A': 11,
      'Ligue 1': 13,
      'La Liga': 12,
      'Bundesliga': 20
    }

    self.STAT_TYPES = [
      'standard',
      'keeper', 
      'keeper_adv', 
      'shooting', 
      'passing', 
      'passing_types', 
      'goal_shot_creation', 
      'defense', 
      'possession',
      'playing_time',
      'misc'
    ]

    self.no_cache = no_cache
    self.timeout = timeout

  def get_leagues(self):
    df = self.fbref.read_leagues()
    self.save_data_to_csv(df, 'data', 'leagues.csv')

    return df
  
  ### Player Season Stats ###
  def get_player_season_stats(self):
    stat_types = ['standard']

    for stat in stat_types:
      # Mutes warning logs from Soccerdata
      with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        # url = f"https://fbref.com/en/comps/Big5/{self.season}/stats/players/{self.season}-Big-5-European-Leagues-Stats"
        df = self.fbref.read_player_season_stats(stat)
        
        for league in df.index.get_level_values('league').unique():
          for season in df.loc[league].index.get_level_values('season').unique(): 
            for team in df.loc[league, season].index.get_level_values('team').unique():
              team_df = df.loc[league, season, team]
              team_df = team_df.reset_index()
              team_df.rename(columns={'index': 'Player'}, inplace=True)

              # Replace '' with NaN in the MultiIndex levels
              team_df.columns = pd.MultiIndex.from_tuples(
                [('General', a.capitalize()) if b == '' else (a.capitalize(), b.capitalize()) for a, b in team_df.columns]
              )

              self.create_or_update_csv(team_df, f"data/{self.season}/leagues/{league}/player_season_stats/{team}", f"{stat}.csv")
      
    return df
  
  ### CUSTOM SCRAPPERS ###

  def get_player_match_logs(self, debug=False):
    season_str = self.format_season_str(self.season)
    player_ids_df = self.get_players_ids()

    players_df = pd.DataFrame()

    for _, player in player_ids_df.iterrows():
      player_name = player['name']
      player_id = player['id']
      player_pos = player['position']

      normalized_name = self.normalize_name(player_name).replace(" ", "-")
      
      subdirectory = f"data/{self.season}/players_match_logs/"
      filename = f"{normalized_name}_{player_id}.csv"
      script_dir = os.path.dirname(os.path.abspath(__file__))
      file_path = os.path.join(script_dir, subdirectory, filename)

      if os.path.exists(file_path):
        if debug:
          print(f"Loading: {file_path}")
        player_df = pd.read_csv(file_path, header=[0])
      else:
        if not player_pos == 'GK':
          stat_type = 'summary'
        else: 
          stat_type = 'keeper'

        url = f'https://fbref.com/en/players/{player_id}/matchlogs/{season_str}/{stat_type}/{normalized_name}-Match-Logs'

        if debug:
          print(f"Fetching: {url}")

        response = requests.get(url)

        dfs = pd.read_html(url, extract_links='all')
        player_df = dfs[0]

        match_data = player_df.iloc[:, 0]

        player_df = player_df.iloc[:, 1:]
        player_df.columns = [self.reformat_column_name(col) for col in player_df.columns]
        
        # TODO: Remove other cols
        player_df = player_df.drop(columns=['day', 'venue', 'pos', 'match report'])

        # Reformat every cell in df
        for col in player_df.columns:
          if col in ['squad', 'opponent']:
            player_df[col] = player_df[col].apply(self.extract_team_id)
          else:
            player_df[col] = player_df[col].apply(self.extract_first_value)

        # Format start value
        def convert_start_value(x):
          return 1 if x == "Y" or x == "Y*" else 0 if x == "N" else x
        player_df['start'] = player_df['start'].apply(convert_start_value)

        # Adds the match data
        player_df['date'] = match_data.apply(lambda x: x[0] if isinstance(x, tuple) else x)

        def extract_match_id(x):
          if isinstance(x, tuple) and x[1] is not None:
            match = re.search(r"en/matches/([a-fA-F0-9]+)", x[1])
            return match.group(1) if match else None
          return x
        player_df['match_id'] = match_data.apply(extract_match_id)  

        # Filter out data thats not from league competitions
        player_df = player_df[player_df['comp'].isin(self.LEAGUES.keys())]

        # Drop rows with Nan data
        player_df = player_df.dropna()
        
        self.save_data_to_csv(player_df, subdirectory, filename)
        time.sleep(self.timeout)

      if not 'id' in player_df.columns:
        player_df['id'] = player_id

      players_df = pd.concat([players_df, player_df], ignore_index=True)
    
    return players_df

  # Extracts the id from tuple second item
  def extract_team_id(self, x):
    if isinstance(x, tuple) and x[1] is not None:
      match = re.search(r"en/squads/([a-fA-F0-9]+)", x[1])
      return match.group(1) if match else x[1]

  # Extracts the first value in each tuple
  def extract_first_value(self, x):
    if isinstance(x, tuple) and x[0] is not None:
      if x[0] == 'On matchday squad, but did not play':
        return 0

      match = re.search(r"\((\d+),?\)", x[0])
      return float(match.group(1)) if match else x[0]

  def reformat_column_name(self, col):
    """Formats column names based on the given tuple structure."""
    
    if isinstance(col, tuple) and len(col) == 2:
      first, second = col
      if first == ('', None):
        return second[0].lower()
      elif isinstance(first, tuple) and isinstance(second, tuple):
        return f"{first[0].lower()}_{second[0].lower()}"
      return col[0]

  # Gets FBref player ids
  def get_players_ids(self, debug=False):
    season_teams = self.team_matcher.get_season_teams(self.season)

    players_df = pd.DataFrame()

    for key in season_teams:
      team = season_teams[key]
      fbref_id = team['FBref']['id']
      team_url_name = team['FBref']['url_name']

      subdirectory = f'data/{self.season}/team_ids'
      filename = f"{team['FBref']['name']}.csv"
      script_dir = os.path.dirname(os.path.abspath(__file__))
      file_path = os.path.join(script_dir, subdirectory, filename)

      if os.path.exists(file_path):
        if debug:
          print(f"Loading: {file_path}")
        team_df = pd.read_csv(file_path, header=[0])
        team_df.columns = [col[0] if isinstance(col, tuple) else col for col in team_df.columns]
      else:
        season_str = self.format_season_str(self.season)
        url = f"https://fbref.com/en/squads/{fbref_id}/{season_str}/{team_url_name}-Stats"
        
        if debug:
          print(f"Fetching: {url}")

        response = requests.get(url)

        dfs = pd.read_html(url, extract_links='all')
        df = dfs[0]

        players_data = df.iloc[:, 0]
        positions = df.iloc[:, 2]

        team_df = pd.DataFrame(players_data.tolist(), columns=['name', 'url'])
        team_df['position'] = positions.apply(lambda x: x[0] if isinstance(x, tuple) else x)

        team_df['id'] = team_df['url'].str.extract(r"en/players/([a-fA-F0-9]+)")[0]
        team_df = team_df.drop(columns=['url'])
        team_df = team_df.dropna(subset=['id'])

        self.save_data_to_csv(team_df, subdirectory, filename)
        time.sleep(self.timeout)
      
      players_df = pd.concat([players_df, team_df], ignore_index=True)

    players_df = players_df.drop_duplicates(subset=['id'], keep='first')
    return players_df

  def get_league_stats(self, leagues=None, debug=False):
    season_str = self.format_season_str(self.season)
    leagues_team_df = pd.DataFrame()

    available_leagues = self.LEAGUES
    if leagues:
      if isinstance(leagues, str):
        leagues = [leagues]

      invalid = [l for l in leagues if l not in available_leagues]
      if invalid:
        raise Exception(f"Invalid league(s): {invalid}. Options: {list(available_leagues.keys())}")

      selected_leagues = {k: available_leagues[k] for k in leagues}
    else:
      selected_leagues = available_leagues
    
    for league, league_code in selected_leagues.items():
      subdirectory = f'data/{self.season}/leagues'
      filename = f"{league}.csv"
      script_dir = os.path.dirname(os.path.abspath(__file__))
      file_path = os.path.join(script_dir, subdirectory, filename)

      if os.path.exists(file_path):
        if debug:
          print(f"Loading: {file_path}")
        league_df = pd.read_csv(file_path, header=[0])

      else:
        league_url = league.replace(' ', '-')
        url = f"https://fbref.com/en/comps/{league_code}/{season_str}/{season_str}-{league_url}-Stats"
        
        if debug:
          print(f"Fetching: {url}")

        dfs = pd.read_html(url, extract_links='all')
        league_df = dfs[0]

        league_df = league_df.iloc[:, 0:13].copy()
        league_df.columns = [self.reformat_column_name(col) for col in league_df.columns]

        for col in league_df.columns:
          if col in ['squad']:
            league_df[col] = league_df[col].apply(self.extract_team_id)
          else:
            league_df[col] = league_df[col].apply(self.extract_first_value)

        self.save_data_to_csv(league_df, subdirectory, filename)
        time.sleep(self.timeout)
      
      league_df["league"] = league
      leagues_team_df = pd.concat([leagues_team_df, league_df], ignore_index=True)
    
    leagues_team_df = leagues_team_df.rename(columns={
      'Squad': 'team',
      'Rk': 'position'
    })
    return leagues_team_df

  def save_df_to_csv(self, current_df: pd.DataFrame, subdirectory, filename):
    # Get the directory of the current script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path for the nested subdirectory
    dir_path = os.path.join(script_dir, subdirectory)

    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, filename)

    df = pd.DataFrame()

    if (os.path.isfile(file_path)):
      df = df.merge(pd.read_csv(dir_path))

  def extract_player_id(self, cell: tuple):
    pattern = r'/players/([^/]+)/'

    match = re.search(pattern, cell[1])

    return match.group(1)
  
  # Extracts first item in tuple
  def extract_first_element(self, cell):
    if isinstance(cell, tuple):
      return cell[0]
    return cell 

  ### Utils ###
  def save_data_to_csv(self, df: pd.DataFrame, subdirectory, filename):
    # Get the directory of the current script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path for the nested subdirectory
    full_path = os.path.join(script_dir, subdirectory)

    # Create the subdirectory if it doesn't exist
    os.makedirs(full_path, exist_ok=True)

    # Construct the full file path
    file_path = os.path.join(full_path, filename)

    df.to_csv(file_path, index=False)

  def create_or_update_csv(self, df: pd.DataFrame, subdirectory: str, filename: str):
    # Construct the full file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, subdirectory, filename)

    if os.path.exists(file_path):
      # Load the existing CSV file into a DataFrame
      existing_df = pd.read_csv(file_path, header=[0, 1])
      
      # Align the columns of the new DataFrame with the existing DataFrame
      updated_df = existing_df.copy()
      
      for col in df.columns:          
        updated_df[col] = df[col]

      # Nullify any empty string values
      updated_df.replace('', np.nan, inplace=True)
      
      # Save the updated DataFrame back to the CSV
      updated_df.to_csv(file_path, index=False)
    else:
      # If the file does not exist, use the save_data_to_csv method
      self.save_data_to_csv(df, subdirectory, filename)
  
  def format_season_str(self, season_str):
    """Converts a season string from 'YYYY-YY' to 'YYYY-YYYY' format if necessary."""
    start, end = season_str.split('-')
    
    # If the end part already has 4 digits, return as is
    if len(end) == 4:
      return season_str

    return f"{start}-{start[:2]}{end}"

  def normalize_name(self, name):
    # Normalize Unicode characters
    normalized = unicodedata.normalize('NFKD', name)
    # Encode to ASCII, ignore non-ASCII characters
    ascii_name = normalized.encode('ascii', 'ignore').decode('utf-8')
    return ascii_name
