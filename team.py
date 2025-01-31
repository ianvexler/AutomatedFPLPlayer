from itertools import chain
import random

class Team:
  def __init__(self, data):
    self.data = data
    self.current_team = {
      'GK': [],
      'DEF': [],
      'MID': [],
      'FWD': []
    }
    self.current_gw = 1

    # Update to expected points
    self.TARGET = 'total_points'
    self.POSITIONS = ['GK', 'DEF', 'MID', 'FWD']
    self.POS_DISTRIBUTION = {
      'GK': 2,
      'DEF': 5,
      'MID': 5,
      'FWD': 3
    }
    self.MAX_BUDGET = 1000

    # To be used later
    self.gw = 1

  def initial_team(self):
    # TODO: Move the else to a separate function
    # if self.current_gw == 1:
    #   self.budget = 100
    # else:
    #   self.budget = self.team_value() + self.budget

    budget = self.MAX_BUDGET

    best_team, team_cost, team_points = self.get_best_team(budget)

    print(f"Team Cost: {team_cost}")
    print(f"Team Points: {team_points}")

    return best_team

  # Selects the initial team trying to optimize selection
  def get_best_team(self, budget):
    print('Selecting initial team')
  
    # Max available to spend per position
    team_split = {
      'GK': 0.1,
      'DEF': 0.3,
      'MID': 0.4,
      'FWD': 0.2
    }

    pos_budgets = {}
    best_team = {}

    for position in self.POSITIONS:
      # Initializes dict with empty position
      best_team[position] = []

      # Initailizes budgets per position dict
      pos_budgets[position] = budget * team_split[position]

    team_cost = 0
    team_points = 0
    team_distribution = self.POS_DISTRIBUTION.copy()

    while any(value > 0 for value in team_distribution.values()):
      # Get only keys with non-zero values
      available_keys = [key for key, value in team_distribution.items() if value > 0]
      
      if available_keys:
        position = random.choice(available_keys)
        team_distribution[position] -= 1

      # Get best player of random position selected
      pos_min = self._pos_min_price(position)
      pos_count = team_distribution[position]

      # How much can be spent in the next player
      player_budget = pos_budgets[position] - (pos_min * pos_count)
    
      player_id, player_cost, player_points = self._get_best_player(position, player_budget, best_team)
      # Adds player to best_team
      best_team[position].append(player_id)

      # Updates total budget available for position
      pos_budgets[position] -= player_cost
      team_cost += player_cost
      team_points += player_points

    return best_team, team_cost, team_points

  # TODO: Calculate team value
  def team_value():
    return self.budget

  # Returns all the teams which already have 3 players selected
  def _get_teams_capped(self, current_team):
    team_count = {}
    current_team_list = list(chain(*current_team.values()))
    
    for player in current_team_list:
      player_details = self.data[self.data['id'] == player].iloc[0]
      player_team = player_details['team']

      team_count.setdefault(player_team, 0)
      team_count[player_team] += 1
  
    teams_capped = [key for key, value in team_count.items() if value == 3]
    return teams_capped

  # Returns the best player available given postion and budget
  # Returns id, cost
  def _get_best_player(self, position, budget, current_team):
    selected_players = current_team[position]
    teams_capped = self._get_teams_capped(current_team)
    
    # TODO: Get players available given restrictions, budget and position
    players_available = self.data[
      (self.data['position'] == position) & 
      (self.data['cost'] <= budget) &
      (~self.data['id'].isin(selected_players)) &
      (~self.data['team'].isin(teams_capped))
    ]
    
    # Sort players avaialble by points
    players_available = players_available.sort_values(by=[self.TARGET], ascending=False)

    best_player = players_available.iloc[0]

    return best_player['id'], best_player['cost'], best_player['total_points']

  def _pos_min_price(self, position):
    """
    Returns the minimum price for a player in a given position.

    Parameters:
        - position (str): The position of the player ('GK', 'DEF', 'MID', 'FWD').

    Returns:
        - float: The minimum price for a player in the position.
    """
    if position == 'GK' or position == 'DEF':
        return 40
    elif position == 'MID' or position == 'FWD':
        return 45
    else:
        print(f'Invalid position: {position}')
