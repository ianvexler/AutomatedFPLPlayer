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
    self.TEAM_FEATURES = ['strength', 'strength_overall_home', 
      'strength_overall_away',	'strength_attack_home',	'strength_attack_away',	
      'strength_defence_home', 'strength_defence_away'
    ]
    self.GW_TEAM_FEATURES = ['home_team_strength', 'home_team_strength_overall_home', 
      'home_team_strength_overall_away', 'home_team_strength_attack_home', 'home_team_strength_attack_away', 
      'home_team_strength_defence_home', 'home_team_strength_defence_away', 'away_team_strength', 
      'away_team_strength_overall_home', 'away_team_strength_overall_away', 'away_team_strength_attack_home', 
      'away_team_strength_attack_away', 'away_team_strength_defence_home', 'away_team_strength_defence_away'
    ]

    self.ADDITIONAL_FEATURES = ['was_home', 'cost']

    self.CUSTOM_FEATURES = ['gw_decay']

    self.features = {
      'GK': ['saves', 'penalties_saved', 'clean_sheets', 'goals_conceded', 'total_points'],
      'DEF': ['assists', 'goals_scored', 'clean_sheets', 'total_points', 'goals_conceded'],
      'MID': ['assists', 'goals_scored', 'total_points'],
      'FWD': ['assists', 'goals_scored', 'total_points']
    }

    self.TARGET = 'total_points'
    self.EXPECTED = 'expected_points'
    self.BASELINE = 'xP'
    self.COST = 'cost'

  def get_features_for_position(self, position, include_prev_season=False, include_season=False, include_fbref=False):
    # Initialize features
    features = self.GW_TEAM_FEATURES + self.ADDITIONAL_FEATURES + self.CUSTOM_FEATURES

    # Include previous season's features if requested
    if include_prev_season:
      season_features = self.position_season_features(position)
      prev_season_features = [f'prev_season_{feature}' for feature in season_features]
      features += prev_season_features

    # Choose between FBref or default FPL features
    if include_fbref:
      fbref_features = self.position_fbref_features(position)
      features += fbref_features + self.TARGET  # Ensure target is included
    else:
      features += self.features[position]  # Default FPL features

    return features


  def position_fbref_features(self, position):
    match position:
      case 'GK':
        return [
          "performance_cs", "performance_saves", "performance_ga", "minutes",
          "penalty kicks_pksv", "performance_psxg", "performance_save%", 
          "goal kicks_launch%", "form", "total_points", "points_per_game"
        ]
      case 'DEF':
        return [
          "performance_tkl", "performance_int", "performance_blocks", "performance_crdy",
          "performance_gls", "performance_ast", "expected_xg", "expected_xag",
          "sca_sca", "carries_prgc", "form", "total_points", "points_per_game", "yellow_cards", "threat"
        ]
      case 'MID':
        return [
          "performance_gls", "performance_ast", "expected_xg", "expected_xag",
          "sca_sca", "sca_gca", "passes_cmp%", "passes_prgp",
          "carries_carries", "carries_prgc", "take-ons_att", "take-ons_succ",
          "form", "total_points", "points_per_game", "yellow_cards", "threat"
        ]
      case 'FWD':
        return [
          "performance_gls", "performance_ast", "performance_sh", "performance_sot",
          "expected_xg", "expected_npxg", "expected_xag", "sca_sca", "sca_gca",
          "carries_prgc", "take-ons_att", "take-ons_succ",
          "form", "total_points", "points_per_game", "yellow_cards", "threat"
        ]
      case _:
        raise Exception(f"Invalid position provided: {position}")

    
  def position_season_features(self, position):
    match position:
      case 'GK':
        # return [
        #   "clean_sheets", "saves", "expected_goals_conceded", "goals_conceded", "minutes",
        #   "penalties_saved", "bonus", "bps", "influence", "expected_goals_conceded_per_90",
        #   "saves_per_90", "clean_sheets_per_90", "starts", "red_cards", "yellow_cards", "form",
        #   "total_points", "points_per_game", "starts_per_90"
        # ]
        return [
          "clean_sheets", "saves", "goals_conceded", "minutes",
          "penalties_saved", "influence", "yellow_cards", "form",
          "total_points", "points_per_game"
        ]
      case 'DEF':
        # return [
        #   "clean_sheets", "goals_conceded", "expected_goals_conceded", "minutes", "goals_scored",
        #   "assists", "bonus", "bps", "influence", "creativity", "expected_assists",
        #   "expected_goals", "expected_goal_involvements", "clean_sheets_per_90", "expected_goals_conceded_per_90",
        #   "goals_conceded_per_90", "expected_assists_per_90", "expected_goals_per_90",
        #   "expected_goal_involvements_per_90", "own_goals", "starts", "starts_per_90",
        #   "form", "total_points", "points_per_game", "yellow_cards", "red_cards", "threat"
        # ]
        return [
          "clean_sheets", "goals_conceded", "minutes", "goals_scored",
          "assists", "influence", "creativity", "form", "total_points", 
          "points_per_game", "yellow_cards", "threat"
        ]
      case 'MID':
        # return [
        #   "goals_scored", "assists", "bonus", "bps", "influence", "creativity", "expected_assists",
        #   "expected_goals", "expected_goal_involvements", "minutes", "goals_conceded",
        #   "expected_goals_conceded", "goals_conceded_per_90", "expected_goals_per_90", "expected_assists_per_90",
        #   "expected_goal_involvements_per_90", "clean_sheets", "clean_sheets_per_90", "penalties_missed",
        #   "own_goals", "starts", "starts_per_90", "form", "total_points", "points_per_game",
        #   "yellow_cards", "red_cards", "threat", "ict_index"
        # ]
        return [
          "goals_scored", "assists", "influence", "creativity", "minutes", "goals_conceded",
          "clean_sheets", "form", "total_points", "points_per_game",
          "yellow_cards", "threat"
        ]
      case 'FWD':
        # return [
        #   "goals_scored", "assists", "bonus", "bps", "influence", "creativity", "expected_assists",
        #   "expected_goals", "expected_goal_involvements", "minutes", "goals_conceded", "expected_goals_conceded",
        #   "goals_conceded_per_90", "expected_goals_per_90", "expected_assists_per_90",
        #   "expected_goal_involvements_per_90", "penalties_missed", "own_goals", "starts", "starts_per_90",
        #   "form", "total_points", "points_per_game", "yellow_cards", "red_cards", "threat", "ict_index"
        # ]
        return [
          "goals_scored", "assists", "influence", "creativity", "minutes", 
          "form", "total_points", "points_per_game", "yellow_cards", "threat"
        ]
      case _:
        raise Exception(f"Invalid position provided: {position}")
