import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

class FeatureSelector:
  def __init__(self):
    self.MAX_GW = 38

    base_team_features = [
      'strength',
      'strength_overall_home',
      'strength_overall_away',
      'strength_attack_home',
      'strength_attack_away',
      'strength_defence_home',
      'strength_defence_away'
    ]
    self.TEAM_FEATURES = [f"{prefix}_{feature}" for prefix in ['team', 'opponent_team'] for feature in base_team_features]

    self.ADDITIONAL_FEATURES = ['was_home', 'cost']

    self.CUSTOM_FEATURES = ['gw_decay']

    self.features = {
      'GK':  ['saves', 'penalties_saved', 'clean_sheets', 'goals_conceded', 'total_points'],
      'DEF': ['assists', 'goals_scored', 'clean_sheets', 'total_points', 'goals_conceded', 'minutes'],
      'MID': ['assists', 'goals_scored', 'total_points', 'minutes'],
      'FWD': ['assists', 'goals_scored', 'total_points', 'minutes']
    }

    self.TARGET = 'total_points'
    self.EXPECTED = 'expected_points'
    self.BASELINE = 'xP'
    self.COST = 'cost'

  def get_features_for_position(
    self, 
    position, 
    include_prev_season=False, 
    include_fbref=False, 
    include_season=False, 
    include_season_aggs=False,
    include_teams=False,
    top_n=None
  ):
    if top_n:
      return self.position_top_features(position, top_n)

    # Initialize features
    features = self.ADDITIONAL_FEATURES + self.CUSTOM_FEATURES

    if include_teams:
      features += self.TEAM_FEATURES

    # Include previous season's features if requested
    if include_prev_season:
      season_features = self.position_season_features(position)
      prev_season_features = [f'prev_season_{feature}' for feature in season_features]
      features += prev_season_features

    if include_season_aggs:
      agg_funcs = ['mean']
      for agg in agg_funcs:
        agg_features = []
        for feature in self.features[position]:
          agg_features.append(f"{agg}_{feature}")
        features += agg_features

    # Choose between FBref or default FPL features
    if include_fbref:
      fbref_features = self.position_fbref_features(position)
      features += fbref_features + [self.TARGET]  # Ensure target is included
    else:
      features += self.features[position]  # Default FPL features

    return features

  def position_fbref_features(self, position):
    match position:
      case 'GK':
        return [
          "performance_cs", "performance_saves", "performance_ga", "minutes",
          "penalty kicks_pksv", "performance_psxg", "performance_save%" 
        ]
      case 'DEF':
        return [
          "performance_tkl", "performance_int", "performance_blocks", "performance_crdy",
          "performance_gls", "performance_ast", "expected_xg", "expected_xag",
          "sca_sca", "carries_prgc"
        ]
      case 'MID':
        return [
          "performance_gls", "performance_ast", "expected_xg", "expected_xag",
          "sca_sca", "sca_gca", "passes_cmp%", "passes_prgp",
          "carries_carries", "carries_prgc", "take-ons_att", "take-ons_succ"
        ]
      case 'FWD':
        return [
          "performance_gls", "performance_ast", "performance_sh", "performance_sot",
          "expected_xg", "expected_npxg", "expected_xag", "sca_sca", "sca_gca",
          "carries_prgc", "take-ons_att", "take-ons_succ"
        ]
      case _:
        raise Exception(f"Invalid position provided: {position}")

  def position_season_features(self, position):
    match position:
      case 'GK':
        return [
          "clean_sheets", "saves", "goals_conceded", "minutes",
          "penalties_saved", "influence", "yellow_cards", "form",
          "total_points", "points_per_game"
        ]
      case 'DEF':
        return [
          "clean_sheets", "goals_conceded", "minutes", "goals_scored",
          "assists", "influence", "creativity", "form", "total_points", 
          "points_per_game", "yellow_cards", "threat"
        ]
      case 'MID':
        return [
          "goals_scored", "assists", "influence", "creativity", "minutes", "goals_conceded",
          "clean_sheets", "form", "total_points", "points_per_game",
          "yellow_cards", "threat"
        ]
      case 'FWD':
        return [
          "goals_scored", "assists", "influence", "creativity", "minutes", 
          "form", "total_points", "points_per_game", "yellow_cards", "threat"
        ]
      case _:
        raise Exception(f"Invalid position provided: {position}")

  def position_top_features(self, position, top_n):
    match position:
      case 'GK':
        return ['total_points', 'performance_ga', 'performance_saves', 
          'performance_psxg', 'minutes', 'mean_total_points', 'mean_clean_sheets', 
          'mean_saves', 'mean_goals_conceded', 'team_strength_attack_away', 'cost', 
          'team_strength_defence_home', 'performance_save%', 'opponent_team_strength', 
          'opponent_team_strength_attack_home', 'gw_decay', 'team_strength_overall_away', 
          'opponent_team_strength_attack_away', 'opponent_team_strength_overall_home', 
          'opponent_team_strength_defence_home', 'was_home', 'penalty kicks_pksv', 'mean_penalties_saved', 
          'opponent_team_strength_defence_away', 'team_strength_overall_home', 'team_strength_attack_home', 
          'opponent_team_strength_overall_away', 'team_strength_defence_away', 'team_strength', 
          'performance_cs'][:top_n]
      
      case 'DEF':
        return ['total_points', 'performance_blocks', 'sca_sca', 'performance_int', 'performance_tkl', 
          'mean_total_points', 'team_strength_attack_away', 'cost', 'carries_prgc', 'was_home', 'team_strength_overall_home', 
          'team_strength_defence_away', 'mean_minutes', 'team_strength_overall_away', 'opponent_team_strength_attack_home', 
          'performance_crdy', 'opponent_team_strength_overall_home', 'team_strength', 'opponent_team_strength_overall_away', 
          'opponent_team_strength', 'team_strength_attack_home', 'mean_clean_sheets', 'mean_goals_conceded', 'gw_decay', 
          'opponent_team_strength_defence_home', 'team_strength_defence_home', 'performance_gls', 'performance_ast', 
          'opponent_team_strength_defence_away', 'expected_xag'][:top_n]

      case 'MID': 
        return ['total_points', 'passes_cmp%', 'sca_sca', 'carries_prgc', 'cost', 'mean_total_points', 
          'take-ons_att', 'carries_carries', 'mean_goals_scored', 'take-ons_succ', 'mean_minutes', 
          'opponent_team_strength_attack_away', 'mean_assists', 'team_strength_defence_away', 'passes_prgp', 
          'was_home', 'opponent_team_strength_defence_away', 'team_strength_overall_home', 'expected_xg', 
          'opponent_team_strength_overall_away', 'opponent_team_strength', 'performance_ast', 'opponent_team_strength_attack_home', 
          'performance_gls', 'team_strength_defence_home', 'team_strength_overall_away', 'sca_gca', 'expected_xag', 
          'gw_decay', 'opponent_team_strength_defence_home'][:top_n]
      
      case 'FWD':
        return ['total_points', 'expected_xg', 'sca_sca', 'mean_total_points', 'take-ons_att', 
          'expected_npxg', 'performance_sh', 'performance_sot', 'cost', 'mean_goals_scored', 
          'performance_gls', 'mean_assists', 'team_strength_defence_home', 'carries_prgc', 'take-ons_succ', 
          'was_home', 'team_strength_attack_home', 'opponent_team_strength_overall_home', 
          'team_strength_defence_away', 'opponent_team_strength_defence_home', 'opponent_team_strength_attack_away', 
          'team_strength_overall_home', 'mean_minutes', 'team_strength_attack_away', 'opponent_team_strength', 
          'gw_decay', 'opponent_team_strength_attack_home', 'opponent_team_strength_defence_away', 
          'performance_ast', 'sca_gca'][:top_n]

