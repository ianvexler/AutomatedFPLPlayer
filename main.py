
from team_selector import TeamSelector
from vaastav.data_loader import DataLoader

if __name__=='__main__':
  player_id = '283'
  player_name = 'Mohamed Salah'
  gw_id = '1'

  data_loader = DataLoader()
  data = data_loader.get_players_raw()

  team_selector = TeamSelector(data)
  best_team = team_selector.get_best_team()
  print(best_team)