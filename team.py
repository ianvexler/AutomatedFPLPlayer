from itertools import chain
import random
import numpy as np

class Team:
  def __init__(self, data, teams_data, fixtures_data):
    self.data = data
    self.teams_data = teams_data
    self.fixtures_data = fixtures_data
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

    self.current_team = {}
    for position in self.POSITIONS:
      self.current_team[position] = []

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

    return best_team, team_cost
        
  # Selects the initial team trying to optimize selection
  def get_best_team(self, budget):
    print('Selecting initial team')

    best_team = {}
    best_team_fitness = 0

    for _ in range(1):
      team = {}
      pos_budgets = {}

      team_split = self._random_team_split()

      for position in self.POSITIONS:
        # Initializes dict with empty position
        team[position] = []

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
      
        player_id, player_cost, player_points = self._get_best_player(position, player_budget, team)
        # Adds player to team
        team[position].append(player_id)

        # Updates total budget available for position
        pos_budgets[position] -= player_cost
        team_cost += player_cost
        team_points += player_points
      
      # Check if team is better than current best
      team_fitness = self._calc_team_fitness(team, team_points)

      if team_fitness > best_team_fitness:
        best_team_fitness = team_fitness
        best_team = team

    return best_team, team_cost, team_points

  # TODO: Calculate team fitness using team
  def _calc_team_fitness(self, team, team_points):
    return team_points

  def _calc_player_fitness(self, player, extra_transfers=0, weight_future=0.8, test=True):
    """
    Computes a player's fitness score based on predicted points, price, form, fixtures, 
    playing time consistency, captaincy potential, transfer trends, and additional transfers.

    Returns:
        float: The overall fitness score of the player.
    """    
    # Base Fitness Score
    # Points per million metric (ensures cost-effective selections).
    cost = player['cost']
    expected_points = player[self.TARGET]
    
    if not test:
      return expected_points

    base_fitness = expected_points / cost if cost > 0 else 0
    # print(f"base fitness: {base_fitness}, cost: {cost}, expected: {expected_points}")
    # Transfer Trend Adjustment
    # Players with high net transfers in (bandwagons) get a small boost.
    
    # TODO: Maybe move somewhere else to optimize
    min_balance = self.data['transfers_balance'].min()
    max_balance = self.data['transfers_balance'].max()
    transfers_balance = player['transfers_balance']

    transfer_trend_bonus = (transfers_balance - min_balance) / (max_balance - min_balance)

    # Compute Overall Fitness Score
    fitness_score = (base_fitness + transfer_trend_bonus + expected_points)
    return fitness_score

  # TODO: Have to consider double gameweeks
  def _calc_fixture_fitness(
    team_id, weight_future=0.8, 
    home_bonus=0.5, difficulty_penalty=1.5, difficulty_bonus=1.2
  ):
    """
    Compute fixture-based fitness score, adjusted for the player's team strength.

    Args:
      team_id (float)
      weight_future (float): Decay factor for later fixtures (e.g., 0.8 means GW1 matters more than GW3).
      home_bonus (float): Bonus for home matches.
      difficulty_penalty (float): Penalty per opponent difficulty level (1-5 scale, higher = harder fixture).
      difficulty_bonus (float): Bonus when facing a weaker opponent.

    Returns:
      float: Fixture fitness score.
    """
    team = self._get_team_from_id(team_id)
    team_strength = team['strength']

    upcoming_fixtures = self._get_team_opponents(team_id)

    fixture_fitness = 0

    for i, fixture in upcoming_fixtures.iterrows():
      is_home = int(fixture['team_h']) == team_id
      opponent_strength = fixture["team_a"] if is_home else fixture["team_h"]
      
      adjusted_difficulty = opponent_strength - team_strength

      #  Apply bonus or penalty if opponent is stronger/weaker
      if adjusted_difficulty > 0:
        game_fitness = 5 - (difficulty_penalty * adjusted_difficulty)
      else:
        game_fitness = 5 + (difficulty_bonus * abs(adjusted_difficulty))

      # Apply home advantage
      game_fitness += home_bonus if is_home else 0

      # Weight future fixtures with decay factor
      game_fitness *= (weight_future ** i)

      # Ensure non-negative fitness score
      fixture_fitness += max(0, game_fitness)

    return fixture_fitness

  def _get_team_from_id(self, team_id):
    return self.teams_data[self.teams_data['id'] == team_id]

  # Returns the next 3 oponents for a team
  def _get_team_next_opponents(self, team):
    # Next fixtures for team
    future_team_fixtures = self.fixtures_data[
      self.fixtures_data['team_a'] == team_id |
      self.fixtures_data['team_b'] == team_id
    ]

    # Next 3 gws
    future_team_fixtures = future_team_fixtures[
      (self.fixtures_data['GW'] > self.current_gw) & 
      (self.fixtures_data['GW'] <= (self.current_gw + 3))
    ]

    return future_team_fixtures

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
    players_available = self._get_filtered_players(position, budget, current_team)
    
    # Sort players avaialble by points
    players_available = players_available.sort_values(by=[self.TARGET], ascending=False)
    best_player = players_available.iloc[0]

    return best_player['id'], best_player['cost'], best_player['total_points']

  # TODO: Get players available given restrictions, budget and position
  def _get_filtered_players(self, position, budget, current_team, data=None):
    if data is None or data.empty:
      data = self.data

    selected_players = current_team[position]
    teams_capped = self._get_teams_capped(current_team)

    return data[
      (data['position'] == position) & 
      (data['cost'] <= budget) &
      (~data['id'].isin(selected_players)) &
      (~data['team'].isin(teams_capped))
    ]

  def _add_player_fitness_to_data(self):
    data = self.data.copy()  # Ensure it's a full copy
    data["fitness"] = self.data.apply(lambda player: self._calc_player_fitness(player), axis=1)
    return data

  def perform_transfers(self, current_team, budget, transfers_available):
    possible_transfers = []

    # Apply fitness to all players
    fitness_data = self._add_player_fitness_to_data()
    fitness_data = fitness_data.sort_values(by=['fitness'], ascending=False)

    # Iterates through all players and positions
    for position in self.POSITIONS:
      pos_players = current_team[position]
      
      for player_id in pos_players:
        player = fitness_data[fitness_data['id'] == player_id].iloc[0]
        available_budget = player['cost'] + budget

        available_players = self._get_filtered_players(
          position, 
          available_budget, 
          current_team,
          data=fitness_data
        )
        transfer_option = available_players.iloc[0]
        
        if transfer_option['fitness'] > player['fitness']:
          # print(f"Suggested transfer: {player['id']} -> {transfer_option['id']}")
          fitness_difference = player['fitness'] - transfer_option['fitness']
          possible_transfers.append(((player, transfer_option), fitness_difference))

    sorted_transfers = sorted(possible_transfers, key=lambda x: x[1], reverse=True)

    transfers = sorted_transfers[:transfers_available]    
    return [t[0] for t in transfers]

  # Update to use prices based on data
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


  # TODO: Apply random team split
  def _random_team_split(self, variation=0.2):
    """
    Generates a randomized split of budget across positions while ensuring:
    - The total sum is always 1.
    - Each position gets at least its minimum split.
    - Variation is higher across all positions.
    """
    
    # Base allocation
    base_split = {
      'GK': 0.1,
      'DEF': 0.3,
      'MID': 0.4,
      'FWD': 0.2
    }
    return base_split
