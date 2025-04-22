import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from itertools import chain
import random
import numpy as np
import pandas as pd
from data.data_loader import DataLoader
from utils.team_matcher import TeamMatcher

class TeamManager:
  def __init__(
    self, 
    season, 
    teams_data, 
    fixtures_data, 
    transfers_strategy='simple',
    target='xP'
  ):
    self.season = season
    self.teams_data = teams_data
    self.fixtures_data = fixtures_data

    self.transfers_strategy = transfers_strategy

    # Update to expected points
    self.TARGET = target
    self.OPTIMAL_TARGET = 'total_points'
    self.POSITIONS = ['GK', 'DEF', 'MID', 'FWD']
    
    self.TRANSFER_THRESHOLD = 0.6

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

    self.data_loader = DataLoader(self.season)
        
  # Selects the initial squad trying to optimize selection
  def get_best_squad(self, gw_data, budget, current_gw, optimal=False):
    teams_form = self.calc_teams_form(current_gw)
    
    gw_data = gw_data.copy()
    gw_data["fitness"] = gw_data.apply(lambda player: self._calc_player_fitness(gw_data, current_gw, teams_form, player), axis=1)

    if gw_data.empty:
      return None, None

    best_squad = {}
    best_squad_cost = 0
    best_squad_points = 0

    for _ in range(50):
      squad = {}
      pos_budgets = {}

      squad_split = self._random_squad_split(gw_data)

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

      # TODO: Maybe also include team diversity?
      # squad_points += self.calc_team_diversity(squad)
      
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

  def _calc_player_fitness(
    self, 
    gw_data,
    current_gw,
    teams_form, 
    player,
    points_weight=0.7,
    transfer_weight=0.15,
    fixture_difficulty_weight = 0.15
  ):
    """
    Computes a player's fitness score between 0 and 1 based on:
    - Expected points (main driver)
    - Normalized transfer trends
    - Normalized value (points per cost)
    
    Returns:
        float: Fitness score between 0 (worst) and 1 (best)
    """
    player_id = player['id']
    cost = player['cost'] if player['cost'] > 0 else 1  # avoid divide by 0
    transfers_balance = player.get('transfers_balance', 0)

    # Get expected points for this player
    player_entries = gw_data[gw_data['id'] == player_id]
    expected_points = player_entries[self.TARGET].sum()

    # Normalize expected points across all players
    all_expected_points = gw_data.groupby('id')[self.TARGET].sum()
    min_ep, max_ep = all_expected_points.min(), all_expected_points.max()
    norm_expected_points = (expected_points - min_ep) / (max_ep - min_ep) if max_ep > min_ep else 0

    if self.transfers_strategy == 'simple':
      return norm_expected_points

    # Normalize transfer trend
    min_transfers = gw_data['transfers_balance'].min()
    max_transfers = gw_data['transfers_balance'].max()
    norm_transfer = (transfers_balance - min_transfers) / (max_transfers - min_transfers) if max_transfers > min_transfers else 0

    # Get form of next 3 opponents
    norm_difficulty = self._calc_opponents_form(teams_form, player, current_gw)
    
    # Final weighted fitness score
    fitness_score = (
      norm_expected_points * points_weight +
      norm_transfer * transfer_weight +
      norm_difficulty * fixture_difficulty_weight
    )

    return round(fitness_score, 1)

  # Calculates the form of the next 3 opponents
  def _calc_opponents_form(
    self, 
    teams_form, 
    player, 
    current_gw,
    lambda_decay=0.02
  ):
    team_id = player['team']
    
    # Get your team's form
    team_form = teams_form.loc[teams_form['id'] == team_id, 'form'].iloc[0]

    # Get forms of next opponents    
    future_team_fixtures = self.fixtures_data[
      (self.fixtures_data['team_h'] == team_id) |
      (self.fixtures_data['team_a'] == team_id)
    ]

    # Next 3 gws
    future_team_fixtures = future_team_fixtures[
      (future_team_fixtures['GW'] > current_gw) & 
      (future_team_fixtures['GW'] <= (current_gw + 3))
    ]

    opponent_forms = []

    # Reference kickoff time (current GW)
    gw_fixtures = self.fixtures_data[self.fixtures_data['GW'] == current_gw]
    reference_time = gw_fixtures['kickoff_time'].min()

    for _, fixture in future_team_fixtures.iterrows():
      # Determine opponent and home/away
      if fixture['team_h'] == team_id:
        opponent_id = fixture['team_a']
        adjustment_factor = 1
      else:
        opponent_id = fixture['team_h']
        adjustment_factor = 1.2

      # Calculate days until fixture for GW decay
      days_to = (fixture['kickoff_time'] - reference_time).days
      decay_weight = np.exp(-lambda_decay * days_to)

      # Retrieve opponent form
      opponent_form_row = teams_form[teams_form['id'] == opponent_id]
      if not opponent_form_row.empty:
        opponent_form = opponent_form_row.iloc[0]['form']
        adjusted_opponent_form = opponent_form * adjustment_factor * decay_weight
        opponent_forms.append(adjusted_opponent_form)

    # Calculate the average adjusted opponent form
    avg_opponent_form = np.mean(opponent_forms) if opponent_forms else teams_form['form'].mean()

    # Relative difficulty = difference between opponent's form and team's form
    relative_form_diff = avg_opponent_form - team_form

    # Normalize relative difficulty
    # Compute min/max diffs across the league for normalization
    all_diffs = teams_form['form'].values[:, None] - teams_form['form'].values
    min_diff = np.min(all_diffs)
    max_diff = np.max(all_diffs)

    if max_diff > min_diff:
      norm_difficulty = 1 - (relative_form_diff - min_diff) / (max_diff - min_diff)
    else:
      norm_difficulty = 0.5  # Neutral difficulty if no variation

    return norm_difficulty

  def _get_team_from_id(self, team_id):
    return self.teams_data[self.teams_data['id'] == team_id]

  # Returns a list of ids of the next 3 oponents for a team
  def _get_team_next_opponents(self, team_id, gw):
    # Next fixtures for team
    future_team_fixtures = self.fixtures_data[
      (self.fixtures_data['team_h'] == team_id) |
      (self.fixtures_data['team_a'] == team_id)
    ]

    # Next 3 gws
    future_team_fixtures = future_team_fixtures[
      (future_team_fixtures['GW'] > gw) & 
      (future_team_fixtures['GW'] <= (gw + 3))
    ]

    opponents = []
    for _, fixture in future_team_fixtures.iterrows():
      if fixture['team_h'] == team_id:
        opponents.append(fixture['team_a'])
      else:
        opponents.append(fixture['team_h'])

    return opponents

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
    grouped_players = available_players.groupby('id', as_index=False).agg({
      'fitness': 'first',
      self.TARGET: 'sum',
      self.OPTIMAL_TARGET: 'sum',
      'cost': 'first',
      'name': 'first',
      'position': 'first',
      'team': 'first'
    })

    if self.transfers_strategy == 'simple' or target == self.OPTIMAL_TARGET:
      best_target = target
    else:
      best_target = 'fitness'

    # Sort players based on summed target values 
    grouped_players = grouped_players.sort_values(by=[best_target], ascending=False)

    # Select player with the highest combined points
    return grouped_players.iloc[0]

  # TODO: Get players available given restrictions, budget and position
  def _get_filtered_players(self, gw_data, position, budget, current_team):
    gw_data = gw_data.copy()

    selected_players_ids = current_team[position]['id'].tolist()
    teams_capped = self._get_teams_capped(gw_data, current_team)

    return gw_data[
      (gw_data['position'] == position) & 
      (gw_data['cost'] <= budget) &
      (~gw_data['id'].isin(selected_players_ids)) &
      (~gw_data['team'].isin(teams_capped))
    ].copy()

  # Suggests transfers by replacing underperforming players with those predicted to improve.
  def suggest_transfers(
    self, 
    gw_data, 
    current_gw,
    current_team, 
    budget, 
    force_transfer
  ):
    gw_data = gw_data.copy()
    teams_form = self.calc_teams_form(current_gw)

    # Apply fitness to all players and sort by fitness (best first)
    gw_data["fitness"] = gw_data.apply(lambda player: self._calc_player_fitness(gw_data, current_gw, teams_form, player), axis=1)
    fitness_data = gw_data.sort_values(by=['fitness'], ascending=False)
    
    possible_transfers = []
    
    # Track players already transferred in
    transferred_in = set()

    for position in self.POSITIONS:
      for _, selected_player in current_team[position].iterrows():
        player_row = fitness_data[fitness_data['id'] == selected_player['id']]

        # If the player is missing from the dataset penalize
        if player_row.empty:
          missing_player = self._get_missing_player(selected_player)
          player_row = pd.DataFrame([missing_player])

        player = player_row.iloc[0]

        available_budget = budget + player['cost']

        # Get all valid replacements sorted by fitness
        available_players = self._get_filtered_players(
          fitness_data, position, available_budget, current_team
        ).sort_values(by=['fitness', 'selected', 'agg_total_points', 'cost'], ascending=False)

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
        if self._should_transfer(player_in, player, force_transfer):
          fitness_difference = player_in['fitness'] - player['fitness']

          transferred_in.add(player_in['id'])
          possible_transfers.append({
            'in': player_in,
            'out': player,
            'difference': fitness_difference
          })

    sorted_transfers = sorted(possible_transfers, key=lambda x: x['difference'], reverse=True)
    return sorted_transfers

  def _should_transfer(self, player_in, player_out, force_transfer):
    threshold = self.TRANSFER_THRESHOLD

    # If transfer should be forced, lower threshold
    if force_transfer:
      threshold -= 0.2

    return player_in['fitness'] - player_out['fitness'] >= threshold

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

  def _random_squad_split(self, gw_data, total_budget=1000):
    min_prices = (
      gw_data
      .groupby('position')['cost']
      .min()
      .to_dict()
    )

    # Price-based estimate
    buffer = 0.01
    price_based_split = {
      pos: round((min_prices.get(pos, 40) * count) / total_budget + buffer, 3)
      for pos, count in self.POS_DISTRIBUTION.items()
    }

    # Enforced minimum split
    min_split_floor = {
      'GK': 0.085,
      'DEF': 0.22,
      'MID': 0.25,
      'FWD': 0.22
    }

    min_split = {
      pos: max(price_based_split[pos], min_split_floor[pos])
      for pos in self.POS_DISTRIBUTION
    }

    remaining_budget = 1.0 - sum(min_split.values())
    positions = list(min_split.keys())

    weights = [random.random() * self.POS_DISTRIBUTION[pos] for pos in positions]
    total_weight = sum(weights)

    additional_split = {
      pos: (w / total_weight) * remaining_budget
      for pos, w in zip(positions, weights)
    }

    combined = {
      pos: min_split[pos] + additional_split[pos]
      for pos in positions
    }

    rounded = {pos: round(val, 3) for pos, val in combined.items()}
    total = sum(rounded.values())
    diff = round(1.0 - total, 3)

    if diff != 0:
      remainders = {
        pos: combined[pos] - rounded[pos] for pos in positions
      }
      adjust_pos = max(remainders, key=remainders.get) if diff > 0 else min(remainders, key=remainders.get)
      rounded[adjust_pos] = round(rounded[adjust_pos] + diff, 3)

    return rounded

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
        
        # Gets the predicted GW data for all players in position
        pos_players = gw_data[gw_data['id'].isin(player_ids)]
        pos_players = pos_players.sort_values(by=target, ascending=False)
        
        # Select the players with highest expected points
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

  def calc_team_diversity(self, team, num_bins=11):
    """
    Calculates team diversity as a numeric value between 0 and 1. 
    Based on Gullham's bin method

    Returns:
        float: Diversity score (0 means no diversity, 1 means fully diverse).
    """

    player_costs = [
      player['cost']
      for position in self.POSITIONS
      for _, player in team[position].iterrows()
    ]

    bins = np.linspace(min(player_costs), max(player_costs), num_bins + 1)
    occupied_bins = np.zeros(num_bins, dtype=bool)
    for cost in player_costs:
      bin_index = np.digitize(cost, bins, right=False) - 1
      bin_index = np.clip(bin_index, 0, num_bins - 1)
      occupied_bins[bin_index] = True

    diversity_score = np.sum(occupied_bins)
    return diversity_score

  def calc_teams_form(self, gw, time_steps=5, lambda_decay=0.02):
    teams_form = {
      'id': [],
      'form': []
    }

    fixtures = self.fixtures_data.copy().sort_values(by='kickoff_time')
    fixtures['GW'] = pd.to_numeric(fixtures['GW'], errors='coerce')

    team_ids = set(fixtures['team_h']).union(set(fixtures['team_a']))

    if gw == 1:
      league_stats = self.data_loader.get_league_stats(
        self.season, 
        leagues='Premier League'
      )
      team_matcher = TeamMatcher()

    # Only consider past fixtures
    past_fixtures = fixtures[fixtures['GW'] < gw]

    # Get reference time for time-decay weighting (start of GW)
    gw_fixtures = fixtures[fixtures['GW'] == gw]
    reference_time = gw_fixtures['kickoff_time'].min()

    for team_id in team_ids:
      team_fixtures = past_fixtures[
          (past_fixtures['team_h'] == team_id) | (past_fixtures['team_a'] == team_id)
      ].copy().sort_values(by='kickoff_time').tail(time_steps)

      team_form = 0

      if gw > 1:
        # Compute time decay weights
        days_since = (reference_time - team_fixtures['kickoff_time']).dt.days
        weights = np.exp(-lambda_decay * days_since.to_numpy())

        for idx, (_, fixture) in enumerate(team_fixtures.iterrows()):
          if fixture['team_h'] == team_id:
            team_score = fixture['team_h_score']
            opposition_score = fixture['team_a_score']
            is_home = True
          else:
            team_score = fixture['team_a_score']
            opposition_score = fixture['team_h_score']
            is_home = False

          if pd.isna(team_score) or pd.isna(opposition_score):
            continue 

          match_boost = 1 if is_home else 1.2
          weight = weights[idx]

          if team_score > opposition_score:
            team_form += 3 * match_boost * weight
          elif team_score == opposition_score:
            team_form += 1 * match_boost * weight

      else:
        team_mapping = team_matcher.get_fpl_team(team_id, self.season)
        fbref_team_name = team_mapping['FBref']['name']
        team_stats = league_stats[league_stats['team'] == fbref_team_name].iloc[0]

        team_form = abs(team_stats['position'] - 20) + 1

      teams_form['id'].append(team_id)
      teams_form['form'].append(team_form)

    return pd.DataFrame.from_dict(teams_form)

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