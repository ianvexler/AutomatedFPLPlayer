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
    raw_data = self.get_players_raw()
    
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

    # Set 'id' column as the index and sort the DataFrame
    if 'id' in sanitized_df.columns:
        sanitized_df.set_index('id', inplace=True)
        sanitized_df.sort_index(inplace=True)

    return sanitized_df

  ### ID Dict ###
  def get_id_dict_data(self):
    raw_data = self.get_id_dict()
    ids_df = pd.DataFrame(raw_data)

    return ids_df      

  ### FETCH DATA ###
  def get_players_raw(self):
    players_raw_url = f"{self.season}/players_raw.csv"
    return self.get_csv_data(players_raw_url)
  
  def get_id_dict(self):
    id_dict_url = f"{self.season}/id_dict.csv"
    return self.get_csv_data(id_dict_url)
  
  # Returns the csv data in a np array
  def get_csv_data(self, url):
    response = requests.get(f"{BASE_URL}{url}")
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))

    return df

  ### UTILS ###
  def csv_to_array(self, data):
    csv_reader = csv.reader(io.StringIO(data))
    return np.array(list(csv_reader))
  
  def convert_df_to_numeric(self, df):
    # Apply pd.to_numeric to each cell of the DataFrame
    df = df.map(lambda x: pd.to_numeric(x, errors='coerce'))
    return df






  ### OLD FUNCS ###
  def get_cleaned_players(self):
    cleaned_players_url = f"{self.season}/cleaned_players.csv"
    return self.get_csv_data(cleaned_players_url)
  
  def get_player_list(self):
    player_list_url = self.base_url + "2022-23/player_idlist.csv"
    return self.get_csv_data(player_list_url)
      
  def get_player_data(self, player_id):
    # List of all players
    player_list = self.get_player_list()

    # Finds player according to id and concats name
    player_info = player_list[player_list[:, 2] == player_id][0]
    player_info = '_'.join(player_info)

    # Looks for player's record
    player_url = self.base_url + "2022-23/players/" + player_info + "/gw.csv"
    return self.get_csv_data(player_url)

  def get_gw_data(self, gw_id):
    gw_url = self.base_url + "2022-23/gws/gw" + gw_id + ".csv"
    return self.get_csv_data(gw_url)
  
  def get_understat_data(self, player_id):
    id_dict = self.get_id_dict()

    player_info = id_dict[id_dict[:, 1] == player_id][0]
    player_info = (player_info[2].replace(" ", "_")) + "_" + player_info[0]

    # Looks for player's understat record
    understat_url = f"{self.base_url}2022-23/understat/{player_info}.csv"
    return self.get_csv_data(understat_url)
  
  # ----- Players Raw -----
  def get_raw_index(self, metric):
    players_raw = self.get_players_raw()
    return np.where(players_raw[0] == metric)[0][0]
  
  def get_player_raw(self, player_id):
    players_raw = self.get_players_raw()
    index = np.where(players_raw[0] == 'id')[0][0]
    return players_raw[players_raw[:, index] == player_id][0]
  
  def get_raw_stat(self, metric):
    players_raw = self.get_players_raw()
    index = self.get_raw_index(metric)
    return players_raw[1:, index]
  
  def get_raw_stat_by_pos(self, pos, metric):
    players_raw = self.get_players_raw()
    index = self.get_raw_index("element_type")
    players_raw = players_raw[players_raw[:, index] == pos]
    
    index = self.get_raw_index(metric)    
    return players_raw[:, index]

COMMON_COLS = [
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
  "yellow_cards"
]

SEASON_UNIQUE_COLS = [
  # "chance_of_playing_next_round", # TODO?
  # "chance_of_playing_this_round", # TODO?
  "clean_sheets_per_90",
  # "code",
  # "corners_and_indirect_freekicks_order",
  # "corners_and_indirect_freekicks_text",
  # "cost_change_event",
  # "cost_change_event_fall",
  # "cost_change_start",
  # "cost_change_start_fall",
  # "creativity_rank", # TODO?
  # "creativity_rank_type",
  # "direct_freekicks_order",
  # "direct_freekicks_text",
  # "dreamteam_count",
  "element_type",
  # "ep_next",
  # "ep_this",
  # "event_points",
  "expected_assists_per_90",
  "expected_goal_involvements_per_90",
  "expected_goals_conceded_per_90",
  "expected_goals_per_90",
  # "first_name",
  # "form", # TODO?
  # "form_rank",
  # "form_rank_type",
  "goals_conceded_per_90",
  # "ict_index_rank",
  # "ict_index_rank_type",
  "id",
  # "in_dreamteam",
  # "influence_rank",
  # "influence_rank_type",
  # "news", # TODO?
  # "news_added", # TODO?
  # "now_cost",
  # "now_cost_rank",
  # "now_cost_rank_type",
  # "penalties_order",
  # "penalties_text",
  # "photo",
  "points_per_game",
  # "points_per_game_rank",
  # "points_per_game_rank_type",
  "saves_per_90",
  # "second_name",
  # "selected_by_percent",
  # "selected_rank",
  # "selected_rank_type",
  # "special",
  # "squad_number",
  # "starts_per_90",
  # "status", # TODO?
  "team_code",
  # "threat_rank",
  # "threat_rank_type",
  # "transfers_in_event",
  # "transfers_out_event",
  # "value_form",
  # "value_season",
  # "web_name"
]

DATA_COLUMNS = COMMON_COLS + SEASON_UNIQUE_COLS
