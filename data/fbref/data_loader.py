import soccerdata as sd
import os
import pandas as pd
import re
import numpy as np

class DataLoader:
  def __init__(self, season):
    self.season = season
    self.fbref = sd.FBref(leagues=[
      'Big 5 European Leagues Combined',
      # 'NED-Eredivisie'
    ], seasons=season)

  def get_leagues(self):
    df = self.fbref.read_leagues()
    self.save_data_to_csv(df, 'data', 'leagues.csv')

    return df
  
  ### Player Season Stats ###
  def get_player_season_stats(self, fetch = True):
    if (fetch):
      self.fetch_player_season_stats()

  def fetch_player_season_stats(self):
    for stat in STAT_TYPES:
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
      
      print(f"Fetched and saved {stat}")
  
  ### CUSTOM SCRAPPERS ###
  ### Player Stats Scrapper ###
  def get_player_stats(self):
    # TODO: Loop through all stat types
    stat_type = 'stats'
    
    url = f"https://fbref.com/en/comps/Big5/{self.season}/{stat_type}/players/{self.season}-Big-5-European-Leagues-Stats"

    dfs = pd.read_html(url, extract_links='all')
    df = dfs[0]

    df = self.format_df(df, stat_type)


    # Groups the dfs by league and team
    idx = pd.IndexSlice
    league_series = df.loc[:, idx[:, :, 'League']].squeeze()
    team_series = df.loc[:, idx[:, :, 'Team']].squeeze()

    grouped = df.groupby([league_series, team_series])

    # Saves each df into a df
    for (league, team), team_df in grouped:
      self.save_data_to_csv(team_df, f"data/{self.season}/player_season_stats/leagues/{league}", f"{team}.csv")

    return df
  
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
  
  def format_df(self, df: pd.DataFrame, stat_type):
    column_headers = self.extract_column_headers(df, stat_type)

    df.columns = pd.MultiIndex.from_tuples(column_headers)
    
    df = df.fillna(0)

    players_list = df[(None, None, 'Player')].to_list()
    player_ids = list(map(self.extract_player_id, players_list))

    df = self.add_player_id_to_df(df, player_ids)

    df_formatted = df.map(self.extract_first_element)
    df_formatted = self.format_cols_and_cells(df_formatted)

    return df_formatted

  
  # Function to extract values from a tuple of tuples or a single tuple
  def extract_value(self, input_data):
    extracted_parts = []

    # Check if input_data is a single tuple or a tuple of tuples
    if isinstance(input_data, tuple) and isinstance(input_data[0], tuple):
      # Iterate through each tuple in the tuple of tuples
      for item in input_data:
        if item[1] is None and item[0].strip():
          extracted_parts.append(item[0].strip())
    elif isinstance(input_data, tuple) and len(input_data) == 2:
      # Handle the single tuple case
      if input_data[1] is None and input_data[0].strip():
        extracted_parts.append(input_data[0].strip())

    return extracted_parts

  """
  Formats the columns in a df. E.g.:
    - ('xG', None): 'xG'
    - (('xG', None), ('Per 90 Minutes', None)): 'xG (Per 90 Minutes)'

  Returns:
    List of strings with all formatted cells
  """
  def extract_column_headers(self, df: pd.DataFrame, stat_type):
    columns = []
    
    for col in df.columns:
      values = self.extract_value(col)
        
      if (len(values) == 2):
        column_tuple = (stat_type.capitalize(), values[0], values[1])

      else:
        column_tuple = (None, None, values[0])
        
      columns.append(column_tuple)
        
    return columns
  
  def format_cols_and_cells(self, df):
    df_formatted = df.drop(columns=[
      (np.nan, np.nan, 'Comp'), 
      (np.nan, np.nan, 'Rk'), 
      (np.nan, np.nan, 'Pos'), 
      (np.nan, np.nan, 'Matches')
    ], axis=1)    
    
    df_formatted.insert(6, (np.nan, np.nan, 'Position'), df[(np.nan, np.nan, 'Pos')].str[:2])
    df_formatted.insert(7, (np.nan, np.nan, 'Alt Position'), df[(np.nan, np.nan, 'Pos')].str[3:])
    df_formatted.insert(4, (np.nan, np.nan, 'League'), df[(np.nan, np.nan, 'Comp')].str[3:])
    df_formatted.insert(4, (np.nan, np.nan, 'Team'), df[(np.nan, np.nan, 'Squad')])
    
    df_formatted[(np.nan, np.nan, 'Nation')] = df[(np.nan, np.nan, 'Nation')].str.split(' ').str.get(1)
    
    return df_formatted
  
  def extract_player_id(self, cell: tuple):
    if cell[1] is None:
        return None
    pattern = r'/players/([^/]+)/'

    match = re.search(pattern, cell[1])

    if match:
        return match.group(1)
    else:
        return None
  
  def add_player_id_to_df(self, df, player_id):
    # Tuple where the Player ID is stored using None for the levels
    player_id_tuple = (np.nan, np.nan, 'Player ID')
  
    # Convert player_id to a list if it's a single value for consistency
    if not isinstance(player_id, (list, pd.Series)):
      player_id = [player_id] * len(df)
    
    # Ensure the player_id column exists
    if player_id_tuple in df.columns:
      # Update or assign player_id values
      df[player_id_tuple] = player_id
    else:
      # Add the player_id column for the first time
      df[player_id_tuple] = player_id

    # Drop duplicates based on player_id and keep the last occurrence
    df = df.drop_duplicates(subset=[player_id_tuple], keep='last')
    
    # Drop rows where player_id is None
    df = df.dropna(subset=[player_id_tuple])

    # Reorder columns to make Player ID the third column
    cols = df.columns.tolist()
    if player_id_tuple in cols:
      cols.insert(2, cols.pop(cols.index(player_id_tuple)))
    df = df[cols]
    
    return df
  
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


STAT_TYPES = [
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