from itertools import chain
import random
import numpy as np

class Team:
  def __init__(self, teams_data, fixtures_data):
    self.teams_data = teams_data
    self.fixtures_data = fixtures_data
    self.current_gw = 1

    # Update to expected points
    self.TARGET = 'xP'
    self.OPTIMAL_TARGET = 'total_points'
    self.POSITIONS = ['GK', 'DEF', 'MID', 'FWD']
    self.POS_DISTRIBUTION = {
      'GK': 2,
      'DEF': 5,
      'MID': 5,
      'FWD': 3
    }
    self.MAX_BUDGET = 1000
        
  # Selects the initial squad trying to optimize selection
  def get_best_squad(self, gw_data, budget, optimal=False):
    best_squad = {}
    best_squad_fitness = 0

    for _ in range(100):
      squad = {}
      pos_budgets = {}

      squad_split = self._random_squad_split()

      for position in self.POSITIONS:
        # Initializes dict with empty position
        squad[position] = []

        # Initailizes budgets per position dict
        pos_budgets[position] = budget * squad_split[position]

      squad_cost = 0
      squad_points = 0
      squad_distribution = self.POS_DISTRIBUTION.copy()

      while any(value > 0 for value in squad_distribution.values()):
        # Get only keys with non-zero values
        available_keys = [key for key, value in squad_distribution.items() if value > 0]
        
        if available_keys:
          position = random.choice(available_keys)
          squad_distribution[position] -= 1

        # Get best player of random position selected
        pos_min = self._pos_min_price(position)
        pos_count = squad_distribution[position]

        # How much can be spent in the next player
        player_budget = pos_budgets[position] - (pos_min * pos_count)
      
        player_id, player_cost, player_points = self._get_best_player(gw_data, position, player_budget, squad, optimal)
        
        # Adds player to squad
        squad[position].append(player_id)

        # Updates total budget available for position
        pos_budgets[position] -= player_cost
        squad_cost += player_cost
        squad_points += player_points
      
      # Check if squad is better than current best
      squad_fitness = self._calc_squad_fitness(squad, squad_points)

      if squad_fitness > best_squad_fitness:
        best_squad_fitness = squad_fitness
        best_squad = squad

    return best_squad, squad_cost

  # TODO: Calculate squad fitness using squad
  def _calc_squad_fitness(self, squad, squad_points):
    return squad_points

  def _calc_player_fitness(self, gw_data, player, extra_transfers=0, weight_future=0.8, test=False):
    """
    Computes a player's fitness score based on predicted points, price, form, fixtures, 
    playing time consistency, captaincy potential, transfer trends, and additional transfers.

    Returns:
        float: The overall fitness score of the player.
    """    
    # Base Fitness Score
    # Points per million metric (ensures cost-effective selections).
    cost = player['cost']

    player_entries = gw_data[gw_data['id'] == player['id']]
    expected_points = player_entries[self.TARGET].sum()
    
    if not test:
      return expected_points

    base_fitness = expected_points / cost if cost > 0 else 0

    # Transfer Trend Adjustment
    # Players with high net transfers in (bandwagons) get a small boost.
    
    # TODO: Maybe move somewhere else to optimize
    min_balance = gw_data['transfers_balance'].min()
    max_balance = gw_data['transfers_balance'].max()
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
  def _get_teams_capped(self, gw_data, current_team):
    team_count = {}
    current_team_list = list(chain(*current_team.values()))
    
    for player_id in current_team_list:
      player_details = gw_data[gw_data['id'] == player_id].iloc[0]
      player_team = player_details['team']

      team_count.setdefault(player_team, 0)
      team_count[player_team] += 1
  
    teams_capped = [key for key, value in team_count.items() if value == 3]
    return teams_capped

  # Returns the best player available given postion and budget
  # Returns id, cost
  def _get_best_player(self, gw_data, position, budget, current_squad, optimal=False):
    target = self._get_target(optimal)

    available_players = self._get_filtered_players(gw_data, position, budget, current_squad)
    
    # Group by player ID and sum their total points (handling double gameweeks)
    # grouped_players = available_players.groupby('id', as_index=False).agg({
    #   target: 'sum',
    #   'cost': 'first'
    # })
    grouped_players = available_players

    # Sort players based on summed target values 
    grouped_players = grouped_players.sort_values(by=[target], ascending=False)

    # Select player with the highest combined points
    best_player_id = grouped_players.iloc[0]['id']
    best_player_cost = grouped_players.iloc[0]['cost']
    best_player_points = grouped_players.iloc[0][target]
  
    return best_player_id, best_player_cost, best_player_points


  # TODO: Get players available given restrictions, budget and position
  def _get_filtered_players(self, gw_data, position, budget, current_team):
    selected_players = current_team[position]
    teams_capped = self._get_teams_capped(gw_data, current_team)

    return gw_data[
      (gw_data['position'] == position) & 
      (gw_data['cost'] <= budget) &
      (~gw_data['id'].isin(selected_players)) &
      (~gw_data['team'].isin(teams_capped))
    ]

  def _add_player_fitness_to_data(self, gw_data):
    data = gw_data.copy()  # Ensure it's a full copy
    data["fitness"] = gw_data.apply(lambda player: self._calc_player_fitness(gw_data, player), axis=1)
    return data

  def suggest_transfers(self, gw_data, current_team, budget):
    possible_transfers = []

    # Apply fitness to all players
    fitness_data = self._add_player_fitness_to_data(gw_data)
    fitness_data = fitness_data.sort_values(by=['fitness'], ascending=False)

    # Iterates through all players and positions
    for position in self.POSITIONS:
      pos_players = current_team[position]
      
      for player_id in pos_players:
        player = fitness_data[fitness_data['id'] == player_id].iloc[0]
        available_budget = player['cost'] + budget

        available_players = self._get_filtered_players(
          fitness_data,
          position, 
          available_budget, 
          current_team
        ).sort_values(by=self.TARGET, ascending=False)

        transfer_option = available_players.iloc[0]
        
        if transfer_option['fitness'] > player['fitness']:
          fitness_difference = player['fitness'] - transfer_option['fitness']
          possible_transfers.append({
            'in': transfer_option,
            'out': player, 
            'difference': fitness_difference
          })

    sorted_transfers = sorted(possible_transfers, key=lambda x: x['difference'], reverse=True)
    return sorted_transfers

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
  def _random_squad_split(self, variation=0.2):    
    # Base allocation
    base_split = {
      'GK': 0.1,
      'DEF': 0.3,
      'MID': 0.4,
      'FWD': 0.2
    }
    return base_split

  def should_use_chip(self):
    print('Returns true or false and which chip should be used')

  def select_best_xi(self, selected_team, gw_data, optimal=False):
    target = self._get_target(optimal)

    best_xi = {}
    for position in self.POSITIONS:
      best_xi[position] = []

    # Filter gw_data to only include players in selected_team
    gw_data = gw_data[gw_data['id'].isin(
      selected_team['GK'] + selected_team['DEF'] + selected_team['MID'] + selected_team['FWD']
    )]

    # Using formation 1-3-4-3
    position_limits = {'GK': 1, 'DEF': 3, 'MID': 4, 'FWD': 3}

    for pos, count in position_limits.items():
      pos_players = gw_data[gw_data['id'].isin(selected_team[pos])].sort_values(by=target, ascending=False)
      best_xi[pos] = pos_players['id'].head(count).tolist()

    captain_id = self.select_captain(gw_data, best_xi, optimal)
    
    return best_xi, captain_id

  # Select the player with the highest TARGET value as captain from the selected team
  def select_captain(self, gw_data, best_xi, optimal):
    # Removes GK from captain choices
    best_xi = best_xi.copy()
    best_xi.pop('GK')
    
    target = self._get_target(optimal)
    selected_team_ids = sum(best_xi.values(), [])
    
    # Filter gw_data to only include these players and find the highest TARGET value
    captain = gw_data[gw_data['id'].isin(selected_team_ids)].nlargest(1, self.TARGET)
    return captain['id'].values[0]

  def calc_team_points(self, gw_data, selected_xi, captain_id=None, triple_captain=False):
    total_points = 0
    captain_multiplier = 3 if triple_captain else 2

    for position in self.POSITIONS:
      for player_id in selected_xi[position]:
        try:
          player = gw_data[gw_data['id'] == player_id].iloc[0]
          player_points = player[self.OPTIMAL_TARGET]
        except IndexError:
          player_points = 0
        
        if captain_id == player_id:
          player_points *= captain_multiplier
        total_points += player_points
    
    return total_points

  def _get_target(self, optimal):
    return self.OPTIMAL_TARGET if optimal else self.TARGET
