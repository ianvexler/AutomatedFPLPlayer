
from vaastav.data_loader import DataLoader
from vaastav.gameweek import Gameweek
from vaastav.player import Player

if __name__=='__main__':
  player_id = '283'
  player_name = 'Mohamed Salah'
  gw_id = '1'

  data_loader = DataLoader()

  gw_data = data_loader.get_gw_data(gw_id)
  understat_data = data_loader.get_understat_data(player_id)
  players_raw = data_loader.get_players_raw()

  player_raw = data_loader.get_player_raw(player_id)
  player_data = data_loader.get_player_data(player_id)

  player = Player(name=player_name, data=player_data, raw=player_raw)
  
  gameweek = Gameweek(num=gw_id, data=gw_data)
  print(data_loader.get_raw_stat("bps"))