from data.vaastav.data_loader import DataLoader as Vaastav
from data.sofascore.data_loader import DataLoader as Sofascore

class DataLoader:
  def get_data(self, season):
    data_loader = Vaastav(season)
    data = data_loader.get_full_season_data()

    return data

  def get_id_dict_data(self, season):
    data_loader = Vaastav(season)
    data = data_loader.get_id_dict_data()

    return data

  def get_merged_gw_data(self, season):
    data_loader = Vaastav(season)
    gw_data = data_loader.get_merged_gw_data()

    teams_data = self.get_teams_data(season)

    data = self._add_teams_data_to_gw_data(gw_data, teams_data)

    return data

  def _add_teams_data_to_gw_data(self, gw_data, teams_data):
    home_data = teams_data.rename(columns=lambda x: f"home_team_{x}")
    away_data = teams_data.rename(columns=lambda x: f"away_team_{x}")

    # Merge home team data on 'team' (home team name)
    gw_data = gw_data.merge(
      home_data,
      how='left',
      left_on='team',
      right_on='home_team_name'
    )
    
    # Merge away team data on 'opposition_id' (away team ID)
    gw_data = gw_data.merge(
      away_data,
      how='left',
      left_on='opponent_team',
      right_on='away_team_id'
    )

    team_name_to_id = teams_data.set_index('name')['id'].to_dict()
    gw_data['team'] = gw_data['team'].map(team_name_to_id)
    
    gw_data = gw_data.drop(columns=['home_team_id', 'home_team_name', 'away_team_id', 'away_team_name'], errors='ignore')

    # Move total_points to the first column
    gw_data.insert(0, 'total_points', gw_data.pop('total_points'))

    gw_data.to_csv('test.csv')

    return gw_data

  def get_fixtures(self, season):
    data_loader = Vaastav(season)
    data = data_loader.get_fixtures_data()

    return data

  def get_teams_data(self, season):
    data_loader = Vaastav(season)
    data = data_loader.get_teams_data()

    return data

  def get_players_data(self, season):
    data_loader = Vaastav(season)
    data = data_loader.get_players_data()

    return data
