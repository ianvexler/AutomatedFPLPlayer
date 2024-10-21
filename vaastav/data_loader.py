import io
import numpy as np
import requests
import csv

class DataLoader():
  def __init__(self):
    self.base_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/"

  # Converts csv to np array
  def csv_to_array(self, data):
    csv_reader = csv.reader(io.StringIO(data))
    return np.array(list(csv_reader))
  
  # Returns the csv data in a np array
  def get_csv_data(self, url):
    response = requests.get(url)
    return self.csv_to_array(response.text)
  
  def get_player_list(self):
    player_list_url = self.base_url + "2022-23/player_idlist.csv"
    return self.get_csv_data(player_list_url)
  
  def get_id_dict(self):
    id_dict_url = self.base_url + "2022-23/id_dict.csv"
    return self.get_csv_data(id_dict_url)
  
  def get_cleaned_players(self):
    cleaned_players_url = self.base_url + "2022-23/cleaned_players.csv"
    return self.get_csv_data(cleaned_players_url)
      
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
  def get_players_raw(self):
    players_raw_url = f"{self.base_url}2022-23/players_raw.csv"
    return self.get_csv_data(players_raw_url)
  
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
  