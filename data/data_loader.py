from data.vaastav.data_loader import DataLoader as Vaastav
from data.sofascore.data_loader import DataLoader as Sofascore
from utils.feature_selector import FeatureSelector
from utils.team_matcher import TeamMatcher
from utils.player_matcher import PlayerMatcher
import pandas as pd

class DataLoader:
  def __init__(self):
    self.feature_selector = FeatureSelector()
    self.team_matcher = TeamMatcher()
    self.player_matcher = PlayerMatcher()

  def get_season_data(self, season):
    data_loader = Vaastav(season)
    data = data_loader.get_full_season_data()

    return data

  def get_id_dict_data(self, season):
    data_loader = Vaastav(season)
    data = data_loader.get_id_dict_data()

    return data

  def get_merged_gw_data(self, season, time_steps=0, include_season=True, include_teams=True):
    prev_season = self._decrement_season(season)

    seasons_gw_data = []

    for s in [season, prev_season]:  
      vaastav = Vaastav(s)
      gw_data = vaastav.get_merged_gw_data()
      season_data = vaastav.get_full_season_data()

      if include_teams:
        teams_data = self.get_teams_data(s)
        gw_data = self._add_teams_data_to_gw_data(gw_data, teams_data)

      if include_season:
        season_data = self.get_season_data(s)
        gw_data = self._add_season_data_to_gw_data(gw_data, season_data)

      # Formats data from previous season
      if not s == season:
        relegated_teams = {}

        max_gw = gw_data['GW'].max()
        gw_data = gw_data[gw_data['GW'] > max_gw - time_steps]
        gw_data.loc[:, 'GW'] = gw_data['GW'] - 39

        data_to_remove = []

        # Update player & team IDs to match current season
        for index, player in gw_data.iterrows():
          # Player ID
          player_id = player['id']

          # TODO: Temporary while ID issues remain
          try:
            player_mapping = self.player_matcher.get_fpl_player(player_id, s)
          except:
            data_to_remove.append(index)

          # If player doesnt have current season data
          if season in player_mapping['FPL']:
            # Update player ID
            new_player_id = player_mapping['FPL'][season]['id']
            gw_data.loc[index, 'id'] = new_player_id
            
            # Reassign Team ID
            team_id = player['team']
            team_mapping = self.team_matcher.get_fpl_team(team_id, s)

            # Reassign the id to the team
            # Checks if team got relegated
            if season in team_mapping['FPL']:
              new_team_id = team_mapping['FPL'][season]['id']
              gw_data.loc[index, 'team'] = new_team_id
            else:
              # Assign a new id to relegated teams
              relegated_teams.setdefault(team_id, len(relegated_teams) + 21)
              gw_data.loc[index, 'team'] = relegated_teams[team_id]
          else:
            data_to_remove.append(index)

        gw_data = gw_data.drop(data_to_remove).reset_index(drop=True)
      
      seasons_gw_data.append(gw_data)

    # Concats both seasons data into one df
    merged_data = pd.concat(seasons_gw_data, ignore_index=True)
    merged_data['kickoff_time'] = pd.to_datetime(merged_data['kickoff_time'], errors='coerce') 

    # TODO: Temporary
    # Ensures all players have data for all GWs
    # unique_players = merged_data['id'].unique()
    # players_missing_negative_gw = []

    # for player_id in unique_players:
    #   player_data = merged_data[merged_data['id'] == player_id]
    #   if player_data['GW'].sum() == 38 + time_steps:  # Check if negative GWs exist
    #     players_missing_negative_gw.append(player_id)

    # # If some players are missing negative GWs, add placeholders
    # for player_id in players_missing_negative_gw:
      
    return merged_data.sort_values(by='GW', ascending=True)

  def _fill_missing_gw_data(self, player_id, season_data, prev_season_data):
    player_data = season_data[season_data['id'] == player_id]
    initial_player_data = player_data.sort_values(by='GW', ascending=True).iloc[0]
    initial_value = initial_player_data['value']

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
    season_data = season_data.rename(columns=lambda x: f"prev_season_{x}")
    
    gw_data = gw_data.merge(
      season_data,
      how='left',
      left_on='id',
      right_on='prev_season_id'
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
