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

    self.SEASON_FEATURES = [
      "assists", "bonus", "bps", "clean_sheets", "creativity", "expected_assists", "expected_goal_involvements",
      "expected_goals", "expected_goals_conceded", "goals_conceded", "goals_scored", "ict_index",
      "influence", "minutes", "own_goals", "penalties_missed", "penalties_saved", "red_cards",
      "saves", "starts", "threat", "total_points", "yellow_cards", "clean_sheets_per_90",
      "element_type", "expected_assists_per_90", "expected_goal_involvements_per_90", "expected_goals_conceded_per_90",
      "expected_goals_per_90", "form", "goals_conceded_per_90", "id", "points_per_game", "saves_per_90", "starts_per_90",
    ]

    self.ADDITIONAL_FEATURES = ['was_home', 'value']

    self.features = {
      'GK': ['saves', 'penalties_saved', 'clean_sheets', 'goals_conceded', 'expected_goals_conceded', 'total_points'],
      'DEF': ['assists', 'goals_scored', 'clean_sheets', 'expected_goals', 'expected_assists', 'total_points', 'goals_conceded'],
      'MID': ['assists', 'goals_scored', 'expected_goals', 'expected_assists', 'total_points'],
      'FWD': ['assists', 'goals_scored', 'expected_goals', 'total_points']
    }

  def get_features_for_position(self, position):
    features = self.features[position] + self.GW_TEAM_FEATURES + self.ADDITIONAL_FEATURES

    season_features = self.position_season_features(position)
    season_features = [f'season_{feature}' for feature in season_features]

    return features + season_features

  def get_targets_for_position(self, position):
    return self.features[position]
    
  def position_season_features(self, position):
    match position:
      case 'GK':
        return [
          "clean_sheets", "saves", "expected_goals_conceded", "goals_conceded", "minutes",
          "penalties_saved", "bonus", "bps", "influence", "expected_goals_conceded_per_90",
          "saves_per_90", "clean_sheets_per_90", "starts", "red_cards", "yellow_cards", "form",
          "total_points", "points_per_game", "starts_per_90"
        ]
      case 'DEF':
        return [
          "clean_sheets", "goals_conceded", "expected_goals_conceded", "minutes", "goals_scored",
          "assists", "bonus", "bps", "influence", "creativity", "expected_assists",
          "expected_goals", "expected_goal_involvements", "clean_sheets_per_90", "expected_goals_conceded_per_90",
          "goals_conceded_per_90", "expected_assists_per_90", "expected_goals_per_90",
          "expected_goal_involvements_per_90", "own_goals", "starts", "starts_per_90",
          "form", "total_points", "points_per_game", "yellow_cards", "red_cards", "threat"
        ]
      case 'MID':
        return [
          "goals_scored", "assists", "bonus", "bps", "influence", "creativity", "expected_assists",
          "expected_goals", "expected_goal_involvements", "minutes", "goals_conceded",
          "expected_goals_conceded", "goals_conceded_per_90", "expected_goals_per_90", "expected_assists_per_90",
          "expected_goal_involvements_per_90", "clean_sheets", "clean_sheets_per_90", "penalties_missed",
          "own_goals", "starts", "starts_per_90", "form", "total_points", "points_per_game",
          "yellow_cards", "red_cards", "threat", "ict_index"
        ]
      case 'FWD':
        return [
          "goals_scored", "assists", "bonus", "bps", "influence", "creativity", "expected_assists",
          "expected_goals", "expected_goal_involvements", "minutes", "goals_conceded", "expected_goals_conceded",
          "goals_conceded_per_90", "expected_goals_per_90", "expected_assists_per_90",
          "expected_goal_involvements_per_90", "penalties_missed", "own_goals", "starts", "starts_per_90",
          "form", "total_points", "points_per_game", "yellow_cards", "red_cards", "threat", "ict_index"
        ]
      case _:
        raise Exception(f"Invalid position provided: {position}")
