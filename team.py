from itertools import chain
import random
import numpy as np
import pandas as pd

class Team:
  def __init__(self, teams_data, fixtures_data):
    self.teams_data = teams_data
    self.fixtures_data = fixtures_data
    self.current_gw = 1

    # Update to expected points
    self.TARGET = 'xP'
    self.OPTIMAL_TARGET = 'total_points'
    self.POSITIONS = ['GK', 'DEF', 'MID', 'FWD']
    self.TRANSFER_THRESHOLD = 3

    # TODO: Experiment with differnet positions as captains
    self.CAPTAIN_POSITIONS = ['MID', 'FWD']
    self.POS_DISTRIBUTION = {
      'GK': 2,
      'DEF': 5,
      'MID': 5,
      'FWD': 3
    }
    self.MAX_BUDGET = 1000
    self.FORMATIONS = [
      {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1},  # 5-4-1
      {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2},  # 5-3-2
      {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},  # 4-5-1
      {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},  # 4-4-2
      {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},  # 4-3-3
      {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},  # 3-5-2
      {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},  # 3-4-3
    ]
        
  # Selects the initial squad trying to optimize selection
  def get_best_squad(self, gw_data, budget, optimal=False):
    best_squad = {}
    best_squad_cost = 0
    best_squad_points = 0

    for _ in range(50):
      squad = {}
      pos_budgets = {}

      squad_split = self._random_squad_split()

      for position in self.POSITIONS:
        # Initializes dict with empty position
        squad[position] = pd.DataFrame(columns=gw_data.columns)

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

        target = self._get_target(optimal)
        player = self._get_best_player(gw_data, position, player_budget, squad, target)
        
        # Adds player to squad
        if squad[position].empty:
          squad[position] = pd.DataFrame([player])
        else:
          squad[position] = pd.concat([squad[position], pd.DataFrame([player])], ignore_index=True)

        # Updates total budget available for position
        pos_budgets[position] -= player['cost']
        squad_cost += player['cost']
        squad_points += player[target]
      
      # Check if squad is better than current best
      if squad_points > best_squad_points:
        best_squad_points = squad_points.round(2)

        best_squad = squad
        best_squad_cost = squad_cost

    return best_squad, best_squad_cost

  # Calculate squad fitness using squad
  def calc_squad_fitness(self, gw_data, squad, optimal=False):
    total_points = 0
    target = self._get_target(optimal)

    for position in self.POSITIONS:
      for _, player in squad[position].iterrows():
        player_data = self._get_player_gw_data(player, gw_data)

        total_points += player[target]

    return total_points

  def _calc_player_fitness(self, gw_data, player, weight_future=0.8):
    """
    Computes a player's fitness score based on predicted points, price, form, fixtures, 
    playing time consistency, captaincy potential, transfer trends, and additional transfers.

    Returns:
        float: The overall fitness score of the player.
    """    
    # Base Fitness Score
    cost = player['cost']
    player_entries = gw_data[gw_data['id'] == player['id']]
    expected_points = player_entries[self.TARGET].sum()
    
    base_fitness = expected_points / cost if cost > 0 else 0

    # Transfer Trend Adjustment
    min_balance = gw_data['transfers_balance'].min()
    max_balance = gw_data['transfers_balance'].max()
    transfers_balance = player['transfers_balance']
  
    # Avoid division by zero
    if max_balance > min_balance:
      transfer_trend_bonus = (transfers_balance - min_balance) / (max_balance - min_balance)
    else:
      transfer_trend_bonus = 0

    # Combine Metrics into Final Score
    fitness_score = base_fitness * weight_future + transfer_trend_bonus * (1 - weight_future)

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

  # Returns all the teams which already have 3 players selected
  def _get_teams_capped(self, gw_data, current_team):
    team_count = {}
    
    for pos, df in current_team.items():
      for _, selected_player in df.iterrows():
        player_id = selected_player['id']
        
        try:
          player_details = gw_data[gw_data['id'] == player_id].iloc[0]
          player_team = player_details['team']
        except IndexError:
          player_team = selected_player['team']
        
        team_count.setdefault(player_team, 0)
        team_count[player_team] += 1
          
    teams_capped = [key for key, value in team_count.items() if value == 3]
    return teams_capped

  # Returns the best player available given postion and budget
  # Returns id, cost
  def _get_best_player(self, gw_data, position, budget, current_squad, target):
    available_players = self._get_filtered_players(gw_data, position, budget, current_squad)
    
    # TODO: Handle double GW
    # Group by player ID and sum their total points (handling double gameweeks)
    # grouped_players = available_players.groupby('id', as_index=False).agg({
    #   target: 'sum',
    #   'cost': 'first'
    # })
    grouped_players = available_players

    # Sort players based on summed target values 
    grouped_players = grouped_players.sort_values(by=[target], ascending=False)

    # Select player with the highest combined points
    return grouped_players.iloc[0]

  # TODO: Get players available given restrictions, budget and position
  def _get_filtered_players(self, gw_data, position, budget, current_team):
    selected_players_ids = current_team[position]['id'].tolist()
    teams_capped = self._get_teams_capped(gw_data, current_team)

    return gw_data[
      (gw_data['position'] == position) & 
      (gw_data['cost'] <= budget) &
      (~gw_data['id'].isin(selected_players_ids)) &
      (~gw_data['team'].isin(teams_capped))
    ].copy()

  def _add_player_fitness_to_data(self, gw_data):
    data = gw_data.copy()  # Ensure it's a full copy
    data["fitness"] = gw_data.apply(lambda player: self._calc_player_fitness(gw_data, player), axis=1)
    return data

  # Suggests transfers by replacing underperforming players with those predicted to improve.
  def suggest_transfers(self, gw_data, current_team, budget):
    possible_transfers = []

    # Track players already transferred in
    transferred_in = set()

    # Apply fitness to all players and sort by fitness (best first)
    fitness_data = self._add_player_fitness_to_data(gw_data)
    fitness_data = fitness_data.sort_values(by=['fitness'], ascending=False)

    for position in self.POSITIONS:
      for _, selected_player in current_team[position].iterrows():
        player_row = fitness_data[fitness_data['id'] == selected_player['id']]

        # If the player is missing from the dataset penalize
        if player_row.empty:
          missing_player = self._get_missing_player(selected_player)
          player_row = pd.DataFrame([missing_player])

        player = player_row.iloc[0]

        available_budget = budget + player['cost']

        # Get all valid replacements sorted by self.TARGET
        available_players = self._get_filtered_players(
          fitness_data, position, available_budget, current_team
        ).sort_values(by=self.TARGET, ascending=False)

        # Skip if no valid transfer found
        if available_players.empty:
          continue

        # Find a valid transfer option that hasn't been picked yet
        player_in = None
        for _, candidate in available_players.iterrows():
          if candidate['id'] not in transferred_in:
            player_in = candidate
            break

        # If no valid new player is found, skip
        if player_in is None:
          continue

        # Ensure the new player has a better fitness score
        # if player_in['fitness'] > player['fitness']:
        if self._should_transfer(player_in, player):
          fitness_difference = player_in['fitness'] - player['fitness']

          transferred_in.add(player_in['id'])
          possible_transfers.append({
            'in': player_in,
            'out': player,
            'difference': fitness_difference
          })

    sorted_transfers = sorted(possible_transfers, key=lambda x: x['difference'], reverse=True)
    return sorted_transfers

  def _should_transfer(self, player_in, player_out):
    return player_in['fitness'] - player_out['fitness'] >= self.TRANSFER_THRESHOLD

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
  def _random_squad_split(self):
    base_split = {
      'GK': 0.1,
      'DEF': 0.25,
      'MID': 0.35,
      'FWD': 0.3
    }

    return base_split

  def pick_team(self, selected_squad, gw_data, optimal=False, triple_captain_strat=None):
    target = self._get_target(optimal)

    best_team = None
    max_points = float('-inf')

    # Collect all selected player IDs once
    all_player_ids = {pid for pos in selected_squad for pid in selected_squad[pos]['id']}

    # Identify missing players once
    missing_players = all_player_ids - set(gw_data['id'])
    
    # Generate missing player data once
    if missing_players:
      missing_data = [self._get_missing_player(selected_squad[pos].loc[selected_squad[pos]['id'] == pid].iloc[0])
                      for pos in selected_squad
                      for pid in missing_players if pid in selected_squad[pos]['id'].values]
      gw_data = pd.concat([gw_data, pd.DataFrame(missing_data)], ignore_index=True)

    # Try all formations
    for formation in self.FORMATIONS:
      current_team = {pos: pd.DataFrame() for pos in self.POSITIONS}  # Initialize team
      current_team['bench'] = pd.DataFrame()
      total_points = 0

      for pos, count in formation.items():
        player_ids = selected_squad[pos]['id'].tolist()
        # Select the top players
        pos_players = gw_data[gw_data['id'].isin(player_ids)]
        pos_players = pos_players.sort_values(by=target, ascending=False)
        
        selected_players = pos_players.head(count)
        remaining_players = pos_players.tail(self.POS_DISTRIBUTION[pos] - count)

        current_team[pos] = selected_players
        total_points += selected_players[target].sum()

        current_bench = current_team['bench']
        if current_bench.empty:
          current_team['bench'] = pd.DataFrame(remaining_players)
        else:
          current_team['bench'] = pd.concat([current_team['bench'], pd.DataFrame(remaining_players)])

      # Pick the best captain for this formation, update best team if this formation is better
      if total_points > max_points:
        max_points = total_points

        # Sort bench by target
        current_team['bench'] = current_team['bench'].sort_values(by=target, ascending=False)
        best_team = current_team

    captain, vice_captain = self.select_captains(gw_data, best_team, optimal)

    best_team['captain'] = captain
    best_team['vice_captain'] = vice_captain

    return best_team

  # Select the player with the highest TARGET value as captain from the selected team
  # Returns the id of the selected captain
  def select_captains(self, gw_data, team, optimal):
    current_team = team.copy()
    
    # Get all valid captain choices
    captain_candidates = []
    
    for pos in self.CAPTAIN_POSITIONS:
      if pos in current_team and not current_team[pos].empty:
        captain_candidates.append(current_team[pos])
    captain_candidates = pd.concat(captain_candidates)

    # Get the highest-scoring player based on self.TARGET
    captains = gw_data[gw_data['id'].isin(captain_candidates['id'])].nlargest(2, self.TARGET)

    captain = pd.DataFrame([captains.iloc[0]])
    vice_captain = pd.DataFrame([captains.iloc[1]])

    if captain.empty == None:
      raise Exception('No captain selected')

    return captain, vice_captain

  def calc_team_points(self, gw_data, selected_team, triple_captain=False, bench_boost=False):
    total_points = 0
    captain_multiplier = 3 if triple_captain else 2

    final_team = self._team_after_subs(gw_data, selected_team)

    for position in self.POSITIONS:
      for _, selected_player in final_team[position].iterrows():
        # If player cannot be found assume he didn't play
        try:
          player_id = selected_player['id']
          player = gw_data[gw_data['id'] == player_id].iloc[0]
        except IndexError:
          player = self._get_missing_player(selected_player)

        player_points = player[self.OPTIMAL_TARGET]

        if not final_team['captain'].empty:
          captain = final_team['captain'].iloc[0]
        else:
          captain = None

        if (captain is not None) and (captain['id'] == player_id):
          player_points *= captain_multiplier

        total_points += player_points

    return total_points

  def _team_after_subs(self, gw_data, team):
    available_bench = team['bench']
    new_bench = pd.DataFrame()
    updated_team = {}

    for position in self.POSITIONS:
      updated_team.setdefault(position, pd.DataFrame())

      for _, selected_player in team[position].iterrows():
        try:
          player_id = selected_player['id']
          player = gw_data[gw_data['id'] == player_id].iloc[0]
        except IndexError:
          player = self._get_missing_player(selected_player)

        # If player available in bench
        if (not available_bench.empty) and self._check_player_performed(player):
          # Goalkeepers can only be subbed by goalkeepers
          if position == 'GK':
            sub = available_bench[available_bench['position'] == 'GK'].iloc[0]  
          else:
            suitable_subs = available_bench[available_bench['position'] != 'GK']
            sub = available_bench.iloc[0]

          if self._can_sub(position, team, sub):
            # Remove sub from available bench
            available_bench = available_bench[available_bench['id'] != sub['id']]
            new_bench = self._add_to_df(new_bench, player)
            updated_team[position] = self._add_to_df(updated_team[position], sub)
            continue

        # If sub was not added then add player
        updated_team[position] = self._add_to_df(updated_team[position], player)

    updated_team_ids = set(pd.concat(updated_team.values())['id'])
    
    captain_id = team['captain']['id'].values[0]
    vice_captain_id = team['vice_captain']['id'].values[0]
    
    if captain_id in updated_team_ids:
      new_captain = team['captain']
    elif vice_captain_id in updated_team_ids:
      new_captain = team['vice_captain']
    else:
      new_captain = pd.DataFrame()
    
    updated_team['captain'] = new_captain
    return updated_team

  def _can_sub(self, position, team, sub):
    if position == 'GK' and sub['position'] == 'GK':
      return True
    elif position == 'DEF' and (len(team[position]) > 3 or sub['position'] == 'DEF'):
      return True
    elif position == 'MID' and (len(team[position]) > 3 or sub['position'] == 'MID'):
      return True
    elif position == 'FWD' and (len(team[position]) > 1 or sub['position'] == 'FWD'):
      return True

    return False

  def _check_player_performed(self, player):
    minutes = player['minutes']
    player_points = player[self.OPTIMAL_TARGET]
    
    return minutes == 0 and player_points == 0

  def _get_target(self, optimal):
    return self.OPTIMAL_TARGET if optimal else self.TARGET

  def _get_player_gw_data(self, player, gw_data):
    try:
      player_id = player['id']
      player_data = gw_data[gw_data['id'] == player_id].iloc[0]
    except IndexError:
      player_data = self._get_missing_player(player)

    return player_data

  def _get_missing_player(self, player):
    missing_player = player.to_dict()
    missing_player.update({
      'fitness': 0,
      'minutes': 0,
      self.TARGET: 0,
      self.OPTIMAL_TARGET: 0,
    })
    return missing_player

  def _add_to_df(self, current_df, item):
    item_df = pd.DataFrame([item])

    if current_df.empty:
      return item_df
    else:
      return pd.concat([current_df, item_df], ignore_index=True)