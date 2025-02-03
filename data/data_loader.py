from data.vaastav.data_loader import DataLoader as Vaastav
from data.sofascore.data_loader import DataLoader as Sofascore
from utils.feature_selector import FeatureSelector
import pandas as pd

class DataLoader:
  def __init__(self):
    self.feature_selector = FeatureSelector()

  def get_season_data(self, season):
    data_loader = Vaastav(season)
    data = data_loader.get_full_season_data()

    return data

  def get_id_dict_data(self, season):
    data_loader = Vaastav(season)
    data = data_loader.get_id_dict_data()

    return data

  def get_merged_gw_data(self, season, time_steps, include_season=True, include_teams=True):
    prev_season = self._decrement_season(season)

    seasons_gw_data = []

    for s in [season]:
    # for s in [season, prev_season]:  
      data_loader = Vaastav(s)
      gw_data = data_loader.get_merged_gw_data()
      season_data = data_loader.get_full_season_data()

      if include_teams:
        teams_data = self.get_teams_data(s)
        gw_data = self._add_teams_data_to_gw_data(gw_data, teams_data)

      if include_season:
        season_data = self.get_season_data(s)
        gw_data = self._add_season_data_to_gw_data(gw_data, season_data)

      # Fromats data from previous season
      if not s == season:
        max_gw = gw_data['GW'].max()
        gw_data = gw_data[gw_data['GW'] > max_gw - time_steps]
        gw_data.loc[:, 'GW'] = gw_data['GW'] - 39

      seasons_gw_data.append(gw_data)

    # Concats both seasons data into one df
    merged_data = pd.concat(seasons_gw_data, ignore_index=True)
    merged_data = merged_data.sort_values(by='GW', ascending=True)

    merged_data['kickoff_time'] = pd.to_datetime(merged_data['kickoff_time'], errors='coerce')
    merged_data.to_csv('jhasjdkf.csv')
    return merged_data

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

    return gw_data

  def _add_season_data_to_gw_data(self, gw_data, season_data):
    # season_data = season_data[self.feature_selector.SEASON_FEATURES]
    season_data = season_data.rename(columns=lambda x: f"season_{x}")

    gw_data = gw_data.merge(
      season_data,
      how='left',
      left_on='id',
      right_on='season_id'
    )

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

  def _decrement_season(self, season):
    start_year, end_year = season.split('-')
    new_start = int(start_year) - 1
    new_end = int(end_year) - 1
    return f"{new_start}-{new_end}"
