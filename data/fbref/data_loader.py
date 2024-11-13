import soccerdata as sd
import os
import pandas as pd
import re

class DataLoader:
  def __init__(self, season):
    self.season = season
    self.fbref = sd.FBref(leagues=[
      'Big 5 European Leagues Combined',
      'NED-Eredivisie'
    ], seasons=season)

  def get_leagues(self):
    df = self.fbref.read_leagues()
    self.save_data_to_csv(df, 'data', 'leagues.csv')

    return df
  
  ### Player Season Stats ###
  def get_player_season_stats(self):
    df = self.fbref.read_player_season_stats('standard')

    for league in df.index.get_level_values('league').unique():
      for season in df.loc[league].index.get_level_values('season').unique(): 
        for team in df.loc[league, season].index.get_level_values('team').unique():
            for player in df.loc[league, season, team].index.get_level_values('player').unique():
              player_df = df.loc[league, season, team, player]
              self.save_data_to_csv(player_df, f"data/{self.season}/leagues/{league}/{team}/player_season_stats", f"{player}.csv")
              break
    return df
  
  ### Player Stats Scrapper ###
  def get_player_stats(self):
    url = f"https://fbref.com/en/comps/Big5/{self.season}/stats/players/{self.season}-Big-5-European-Leagues-Stats"

    dfs = pd.read_html(url, extract_links='all')
    df = dfs[0]

    df = self.format_df(df)

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
  
  def format_df(self, df: pd.DataFrame):
    columns = self.format_headers(df)

    df.columns = columns
    df = df.fillna(0)

    players_list = df["Player"].to_list()
    player_ids = list(map(self.extract_player_id, players_list))

    df = self.add_player_id_to_df(df, player_ids)

    df_formatted = df.map(self.extract_first_element)
    df_formatted = self.format_cols_and_cells(df_formatted)

    return df_formatted

  
  # Function to extract values from a tuple of tuples or a single tuple
  def extract_value(input_data):
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
  def format_headers(self, df: pd.DataFrame):
    columns = []
    
    for col in df.columns:            
      values = self.extract_value(col)
        
      if (len(values) == 2):
        column_str = f"{values[1]} ({values[0]})"
      else:
        column_str = values[0]
        
        columns.append(column_str)
        
    return columns
  
  def format_cols_and_cells(df):
    df_formatted = df.drop(columns=['Comp', 'Rk', 'Pos', 'Matches'], axis=1)    
    
    df_formatted.insert(6, 'Position', df['Pos'].str[:2])
    df_formatted.insert(7, 'Alt Position', df['Pos'].str[3:])
    df_formatted.insert(4, 'League', df['Comp'].str[3:])
    
    df_formatted['Nation'] = df['Nation'].str.split(' ').str.get(1)
    
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
    # Convert player_id to a list if it's a single value for consistency
    if not isinstance(player_id, (list, pd.Series)):
        player_id = [player_id] * len(df)
    
    # Ensure the player_id column exists
    if 'Player ID' in df.columns:
        # Update existing player_id entries and ensure uniqueness
        for idx, pid in enumerate(player_id):
            if pid in df['Player ID'].values:
                # Update existing row with the same player_id
                df.loc[df['Player ID'] == pid, :] = df.iloc[idx]
            else:
                # Append new row with unique player_id
                new_row = df.iloc[idx].copy()
                new_row['Player ID'] = pid
                df = df.append(new_row, ignore_index=True)
    else:
        # Add the player_id column for the first time
        df['Player ID'] = player_id
    
    # Drop duplicates based on player_id and keep the last occurrence
    df = df.drop_duplicates(subset=['Player ID'], keep='last')
    
    # Drop rows where player_id is None
    df = df.dropna(subset=['Player ID'])
    
    # Reorder columns to make Player ID the third column
    cols = df.columns.tolist()
    if 'Player ID' in cols:
        cols.insert(2, cols.pop(cols.index('Player ID')))
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