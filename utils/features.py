class Features:
  MAX_GW = 38
  TEAM_FEATURES = ['strength', 'strength_overall_home', 
    'strength_overall_away',	'strength_attack_home',	'strength_attack_away',	
    'strength_defence_home', 'strength_defence_away'
  ]
  GW_TEAM_FEATURES = ['home_team_strength', 'home_team_strength_overall_home', 
    'home_team_strength_overall_away', 'home_team_strength_attack_home', 'home_team_strength_attack_away', 
    'home_team_strength_defence_home', 'home_team_strength_defence_away', 'away_team_strength', 
    'away_team_strength_overall_home', 'away_team_strength_overall_away', 'away_team_strength_attack_home', 
    'away_team_strength_attack_away', 'away_team_strength_defence_home', 'away_team_strength_defence_away'
  ]
  PERFORMANCE_FEATURES = [
    'total_points', 'assists', 'clean_sheets', 'creativity', 'expected_assists', 
    'expected_goal_involvements', 'expected_goals', 'expected_goals_conceded', 
    'goals_conceded', 'goals_scored', 'ict_index', 'influence', 'minutes', 
    'own_goals', 'penalties_missed', 'penalties_saved', 'red_cards', 'saves', 
    'starts', 'threat', 'yellow_cards'
  ]

  @classmethod
  def get_features_for_position(cls, position):
    return cls.position_features(position) + cls.GW_TEAM_FEATURES + ['was_home', 'value']
    
  @classmethod
  def get_targets_for_position(cls, position):
    return cls.position_features(position)

  @classmethod
  def position_features(cls, position):
    match position:
      case 'GK':
        return [
          'total_points', 'clean_sheets', 'expected_goals_conceded', 'goals_conceded', 
          'ict_index', 'influence', 'minutes', 'own_goals', 'penalties_saved', 
          'red_cards', 'saves', 'starts', 'yellow_cards'
        ]
      case 'DEF':
        return [
          'total_points', 'assists', 'clean_sheets', 'creativity', 'expected_assists', 
          'expected_goal_involvements', 'expected_goals', 'expected_goals_conceded', 
          'goals_conceded', 'goals_scored', 'ict_index', 'influence', 'minutes', 
          'own_goals', 'red_cards', 'starts', 'threat', 'yellow_cards'
        ]
      case 'MID':
        return [
          'total_points', 'assists', 'clean_sheets', 'creativity', 'expected_assists', 
          'expected_goal_involvements', 'expected_goals', 'expected_goals_conceded', 
          'goals_conceded', 'goals_scored', 'ict_index', 'influence', 'minutes', 
          'own_goals', 'penalties_missed', 'red_cards', 'starts', 'threat', 'yellow_cards'
        ]
      case 'FWD':
        return [
          'total_points', 'assists', 'clean_sheets', 'creativity', 'expected_assists', 
          'expected_goal_involvements', 'expected_goals', 'expected_goals_conceded', 
          'goals_conceded', 'goals_scored', 'ict_index', 'influence', 'minutes', 
          'penalties_missed', 'red_cards', 'starts', 'threat', 'yellow_cards'
        ]
      case _:
        raise Exception(f"Invalid position provided: {position}")

