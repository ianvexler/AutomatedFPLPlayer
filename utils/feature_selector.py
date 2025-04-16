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
    self.TEAM_FEATURES = [f"{prefix}_{feature}" for prefix in ['home_team', 'away_team'] for feature in base_team_features]

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
    include_teams=False
  ):
    # return self.top_features(position)

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

  def top_features(self, position):
    match position:
      case 'GK':
        return ['total_points', 'mean_clean_sheets', 'performance_ga', 'performance_save%', 'minutes', 'cost', 'performance_psxg', 'mean_saves', 'performance_cs', 'mean_total_points']
      case 'DEF': 
        return ['total_points', 'performance_blocks', 'sca_sca', 'cost', 'mean_total_points', 'performance_int', 'mean_minutes', 'carries_prgc', 'performance_tkl', 'performance_crdy']
      case 'MID': 
        return ['total_points', 'passes_cmp%', 'sca_sca', 'carries_prgc', 'cost', 'mean_total_points', 'take-ons_att', 'carries_carries', 'mean_minutes', 'mean_goals_scored']
      case 'FWD':
        return ['total_points', 'expected_xg', 'cost', 'performance_sh', 'sca_sca', 'take-ons_att', 'mean_goals_scored', 'performance_sot', 'mean_minutes', 'was_home']
      case _:
        raise Exception(f"Invalid position provided: {position}")

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
          "expected_xg", "expected_xag", "sca_sca", "sca_gca",
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
