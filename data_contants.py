TEAMS_COLUMNS = [
  'strength', 'strength_overall_home', 'strength_overall_away',	
  'strength_attack_home',	'strength_attack_away',	'strength_defence_home', 
  'strength_defence_away'
]
GW_TEAMS_COLUMNS = [
  'home_team_strength', 'home_team_strength_overall_home', 'home_team_strength_overall_away',
  'home_team_strength_attack_home', 'home_team_strength_attack_away', 'home_team_strength_defence_home', 
  'home_team_strength_defence_away', 'away_team_strength', 'away_team_strength_overall_home', 
  'away_team_strength_overall_away', 'away_team_strength_attack_home', 'away_team_strength_attack_away', 
  'away_team_strength_defence_home', 'away_team_strength_defence_away'
]
PERFORMANCE_METRICS = [
  'total_points', 'assists', 'clean_sheets', 'creativity', 'expected_assists', 
  'expected_goal_involvements', 'expected_goals', 'expected_goals_conceded', 
  'goals_conceded', 'goals_scored', 'ict_index', 'influence', 'minutes', 
  'own_goals', 'penalties_missed', 'penalties_saved', 'red_cards', 'saves', 
  'starts', 'threat', 'yellow_cards'
]
FEATURE_COLUMNS = PERFORMANCE_METRICS + GW_TEAMS_COLUMNS + ['was_home', 'value']
TARGET_COLUMNS = PERFORMANCE_METRICS
MAX_GW = 38