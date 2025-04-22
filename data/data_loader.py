from data.vaastav.data_loader import DataLoader as Vaastav
from data.sofascore.data_loader import DataLoader as Sofascore
from data.fbref.data_loader import DataLoader as FBref
from data.predictions.data_loader import DataLoader as Predictions
from utils.feature_selector import FeatureSelector
from utils.team_matcher import TeamMatcher
from utils.player_matcher import PlayerMatcher
import pandas as pd
import pickle
from pathlib import Path

import sys 

class DataLoader:
  def __init__(self, no_cache=False, debug=False):
    self.feature_selector = FeatureSelector()
    self.team_matcher = TeamMatcher()
    self.player_matcher = PlayerMatcher()
    self.no_cache = no_cache
    self.debug = debug

    directory = Path("data/cached")
    directory.mkdir(parents=True, exist_ok=True)
    self.CACHE_DIR = directory

  def get_season_data(self, season):
    data_path = self._get_cache_path(f"season_{season}")
    
    # Load if it already exists
    if self._should_load_cached(data_path):
      return self._load_cached_data(data_path)

    data_loader = Vaastav(season)
    data = data_loader.get_full_season_data()

    self._save_cached_data(data_path, data)
    return data

  def get_id_dict_data(self, season):
    data_path = self._get_cache_path(f"id_dict_{season}")
    
    # Load if it already exists
    if self._should_load_cached(data_path):
      return self._load_cached_data(data_path)

    data_loader = Vaastav(season)
    data = data_loader.get_id_dict_data()

    self._save_cached_data(data_path, data)
    return data

  def get_merged_gw_data(
    self, 
    season, 
    time_steps=0, 
    include_prev_season=False, 
    include_fbref=False, 
    include_prev_gws=False, 
    include_season_aggs=False,
    include_teams=False
  ):
    file_name = f"steps_{time_steps}_prev_season_{include_prev_season}_fbref_{include_fbref}_season_aggs_{include_season_aggs}_teams_{include_teams}"
    data_path = self._get_cache_path(f"merged_gw_{season}_{file_name}")
    
    # Load if it already exists
    if self._should_load_cached(data_path):
      return self._load_cached_data(data_path)

    prev_season = self._decrement_season(season)

    seasons_gw_data = []

    seasons = [season]

    if include_prev_gws:
      seasons.append(prev_season)

    for s in seasons:
      vaastav = Vaastav(s)
      gw_data = vaastav.get_merged_gw_data()

      # Include teams data
      if include_teams:
        teams_data = self.get_teams_data(s)
        gw_data = self._add_teams_data_to_gw_data(gw_data, teams_data)

      if include_prev_season:
        season_data = vaastav.get_full_season_data()
        season_data = self.get_season_data(s)
        gw_data = self._add_season_data_to_gw_data(gw_data, season_data)

      # Formats data from previous season
      if not s == season:
        gw_data = self._format_previous_season_gw_data(
          gw_data=gw_data, 
          prev_season=s, 
          current_season=season, 
          time_steps=time_steps
        )

      if include_season_aggs:
        gw_data = self._add_aggs_data_to_gw_data(gw_data)

      # Add FBref match log data
      if include_fbref:
        fbref_data = self.get_fbref_gw_data(s)
        gw_data = self._add_fbref_gw_data_to_gw_data(gw_data, fbref_data, season)

      seasons_gw_data.append(gw_data)

    # Concats both seasons data into one df
    merged_data = pd.concat(seasons_gw_data, ignore_index=True)
    merged_data['kickoff_time'] = pd.to_datetime(merged_data['kickoff_time'], errors='coerce') 

    merged_data = merged_data.sort_values(by='GW', ascending=True)

    self._save_cached_data(data_path, merged_data)
    return merged_data
    
  def get_fbref_gw_data(self, season):
    fbref = FBref(season)
    match_logs_df = fbref.get_player_match_logs()
    
    # TODO: Maybe use match_id for more details?
    match_logs_df = match_logs_df.drop(columns=['comp','round','result','match_id'])

    return match_logs_df

  def _add_fbref_gw_data_to_gw_data(self, gw_data, fbref_data, season):
    merged_data = []

    for _, player_data in gw_data.iterrows():
      player_id = player_data['id']

      # Try and get player ids, if not available, fill with 0s
      try:
        player_ids = self.player_matcher.get_fpl_player(player_id, season)
        fbref_id = player_ids['FBref']['id']
        fbref_player_data = fbref_data[fbref_data['id'] == fbref_id].copy()
        
        fbref_player_data['date'] = pd.to_datetime(fbref_player_data['date']).dt.date
        player_kickoff_date = pd.to_datetime(player_data['kickoff_time']).date()

        # Find matching FBref rows by date
        matching_fbref_data = fbref_player_data[fbref_player_data['date'] == player_kickoff_date]

        # Use FBref data if available, otherwise fill with 0s
        if not matching_fbref_data.empty:
          fbref_stats = matching_fbref_data.iloc[0].drop(labels='id').to_dict()
        else:
          fbref_stats = {col: 0 for col in fbref_data.columns if col != 'id'}  # Fill FBref stats with 0s   
      except:
        fbref_stats = {col: 0 for col in fbref_data.columns if col != 'id'}  # If no FBref ID, fill with 0s

      # Merge player data with FBref stats
      merged_row = {**player_data.to_dict(), **fbref_stats}
      merged_data.append(merged_row)

    return pd.DataFrame(merged_data)

  def _add_teams_data_to_gw_data(self, gw_data, teams_data):
    team_data = teams_data.rename(columns=lambda x: f"team_{x}")
    opponent_data = teams_data.rename(columns=lambda x: f"opponent_team_{x}")

    # Ensure they are the same type
    gw_data['team'] = gw_data['team'].astype(int)
    gw_data['opponent_team'] = gw_data['opponent_team'].astype(int)
    
    team_data['team_id'] = team_data['team_id'].astype(int)
    opponent_data['opponent_team_id'] = opponent_data['opponent_team_id'].astype(int)

    # Merge home team data on 'team' (home team name)
    gw_data = gw_data.merge(
      team_data,
      how='left',
      left_on='team',
      right_on='team_id'
    )
    
    # Merge team data on 'opponent_id'
    gw_data = gw_data.merge(
      opponent_data,
      how='left',
      left_on='opponent_team',
      right_on='opponent_team_id'
    )

    gw_data = gw_data.drop(columns=[
      'team_id', 'team_name', 'opponent_team_id', 'opponent_team_name'
    ], errors='ignore')

    # Move total_points to the first column
    gw_data = gw_data.reindex(columns=['total_points'] + [col for col in gw_data.columns if col != 'total_points'])
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

  # Adds season aggregates up to each GW
  def _add_aggs_data_to_gw_data(self, gw_data):
    agg_funcs = ['mean', 'sum']

    # Iterate through each player's row in gw_data
    for index, player_data in gw_data.iterrows():
      player_id = player_data['id']
      player_pos = player_data['position']
      current_gw = player_data['GW']

      # Get all past gameweeks for this player
      player_gw_data = gw_data[(gw_data['id'] == player_id) & (gw_data['GW'] < current_gw)]

      # Get relevant FPL features
      fpl_features = self.feature_selector.features[player_pos]
      
      if not player_gw_data.empty:
        # Compute aggregates for each feature
        for agg in agg_funcs:
          agg_values = getattr(player_gw_data[fpl_features], agg)()  # Compute mean, sum, etc.

          # Assign aggregate values to the current row
          for feature, value in agg_values.items():
            gw_data.loc[index, f"{agg}_{feature}"] = value  # Add column dynamically
      else:
        # If no past data, fill aggregates with 0
        for agg in agg_funcs:
          for feature in fpl_features:
            gw_data.loc[index, f"{agg}_{feature}"] = 0

    return gw_data    

  def _format_previous_season_gw_data(self, gw_data, prev_season, current_season, time_steps):
    relegated_teams = {}
    max_gw = gw_data['GW'].max()

    # Keep only the most recent 'time_steps' gameweeks
    gw_data = gw_data[gw_data['GW'] > max_gw - time_steps]
    gw_data.loc[:, 'GW'] = gw_data['GW'] - 39

    data_to_remove = []

    # Update player & team IDs to match current season
    for index, player in gw_data.iterrows():
      player_id = player['id']

      try:
        player_mapping = self.player_matcher.get_fpl_player(player_id, prev_season)

        if current_season in player_mapping['FPL']:
          # Update player ID
          new_player_id = player_mapping['FPL'][current_season]['id']
          gw_data.loc[index, 'id'] = new_player_id

          # Update team ID
          team_id = player['team']
          team_mapping = self.team_matcher.get_fpl_team(team_id, prev_season)

          if current_season in team_mapping['FPL']:
            new_team_id = team_mapping['FPL'][current_season]['id']
            gw_data.loc[index, 'team'] = new_team_id
          else:
            relegated_teams.setdefault(team_id, len(relegated_teams) + 21)
            gw_data.loc[index, 'team'] = relegated_teams[team_id]
        else:
          data_to_remove.append(index)

      except Exception as e:
        data_to_remove.append(index)

    gw_data = gw_data.drop(data_to_remove).reset_index(drop=True)
    return gw_data


  def get_fixtures(self, season):
    data_path = self._get_cache_path(f"fixtures_{season}")
    
    # Load if it already exists
    if self._should_load_cached(data_path):
      return self._load_cached_data(data_path)

    data_loader = Vaastav(season)
    data = data_loader.get_fixtures_data()

    self._save_cached_data(data_path, data)
    return data

  def get_teams_data(self, season):
    data_path = self._get_cache_path(f"teams_data_{season}")

    # Load if it already exists
    if self._should_load_cached(data_path):
      return self._load_cached_data(data_path)

    data_loader = Vaastav(season)
    data = data_loader.get_teams_data()

    self._save_cached_data(data_path, data)
    return data

  def get_players_data(self, season):
    data_path = self._get_cache_path(f"players_data_{season}")

    # Load if it already exists
    if self._should_load_cached(data_path):
      return self._load_cached_data(data_path)

    data_loader = Vaastav(season)
    data = data_loader.get_players_data()

    self._save_cached_data(data_path, data)
    return data

  def get_league_stats(self, season, leagues):
    data_path = self._get_cache_path(f"league_stats_{season}_{'_'.join(leagues) if leagues else 'all'}")

    # Load if it already exists
    if self._should_load_cached(data_path):
      return self._load_cached_data(data_path)

    data_loader = FBref(season)
    data = data_loader.get_league_stats(leagues)

    self._save_cached_data(data_path, data)
    return data
  
  def get_gw_predictions(self, season):
    data_loader = Predictions(season)
    df = data_loader.get_gw_predictions()

    # Rename columns
    df = df.rename(columns={'xP': 'fpl_xP', 'expected_points': 'xP'})

    df['GW'] = df['GW'].astype(int)
    df = df.sort_values(['id', 'GW'])

    # Add cumulative total_points per player *excluding current row*
    df['agg_total_points'] = (
      df.groupby('id')['total_points']
        .transform(lambda x: x.cumsum().shift(fill_value=0))
    )

    df['xP'] = df['xP'].round(1)
    return df

  def _decrement_season(self, season):
    start_year, end_year = season.split('-')
    new_start = int(start_year) - 1
    new_end = int(end_year) - 1
    return f"{new_start}-{new_end}"

  def _get_cache_path(self, file_name):
    return self.CACHE_DIR / f"{file_name}.pkl"

  def _load_cached_data(self, data_path):
    if self.debug:
      print(f"Loading cached data from: {data_path}\n")
    
    with open(data_path, "rb") as f:
      return pickle.load(f)

  def _save_cached_data(self, data_path, data):
    with open(data_path, "wb") as f:
      pickle.dump(data, f)

  def _should_load_cached(self, data_path):
    return (not self.no_cache) and data_path.exists()
