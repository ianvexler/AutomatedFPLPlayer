from team import Team
from data.data_loader import DataLoader
import pandas as pd
from enum import Enum
import argparse
import math

class Chip(Enum):
  TRIPLE_CAPTAIN = 'triple_captain'
  WILDCARD = 'wildcard'
  FREE_HIT = 'free_hit'
  BENCH_BOOST = 'bench_boost'

DEFAULT_CONFIG = { 'max_gw': 38 }

class Simulation:
  def __init__(self, season, chip_strategy=None, show_optimal=False, config=DEFAULT_CONFIG):
    self.season = season
    self.POSITIONS = ['GK', 'DEF', 'MID', 'FWD']
    
    self.chips_available = { chip.value: True for chip in Chip }

    data_loader = DataLoader()
    self.fixtures_data = data_loader.get_fixtures(season)
    self.teams_data = data_loader.get_teams_data(season)

    self.team = Team(
      self.teams_data,
      self.fixtures_data
    )

    self.chip_strategy = chip_strategy

    self.MAX_GW = config['max_gw']
    self.MAX_BUDGET = 1000
    self.MAX_FREE_TRANSFERS = 5 if season == '2024-25' else 2
    self.REVAMP_THRESHOLD = 50

    self.cached_best_squad = None
    self.cached_best_squad_cost = None
    self.cached_best_squad_gw = None

    self.total_profit = 0
    self.total_loss = 0

    self.show_optimal = show_optimal
   
    # TODO:
    # - Include option for different strategies
    # - Include multiple risks

  def simulate_season(self):
    print(f"Simulating season {self.season}\n")

    # Ensure variables are restarted
    current_gw = 1
    current_budget = self.MAX_BUDGET
    transfers_available = 0

    total_points = 0

    current_squad = {}
    current_squad_cost = 0

    # Iterates through every GW, including GW1
    for current_gw in range(1, self.MAX_GW + 1):
      if current_gw > 1 and transfers_available < self.MAX_FREE_TRANSFERS:
        transfers_available += 1

      # Restart wildcard on GW20
      if current_gw == 20:
        self.chips_available[Chip.WILDCARD.value] = True

      # Load GW data
      gw_data = self._load_predicted_gw_data(current_gw)
      
      if current_gw == 1:
        # Set initial team for GW1
        current_squad, current_squad_cost = self.team.get_best_squad(gw_data, current_budget)
        current_budget -= current_squad_cost
      
      budget_available = current_budget + current_squad_cost
      use_chip = self.should_use_chip(current_squad, budget_available, gw_data, current_gw)
      
      if use_chip:
        print(f"Using chip: {use_chip.value.replace("_", " ").title()}\n")

      if current_gw > 1:
        if use_chip == Chip.WILDCARD:
          # Adds back the squad cost before using wildcard
          current_budget += current_squad_cost

          # Updates current squads and budget with new cost
          current_squad, current_squad_cost = self.use_wildcard()
          current_budget -= current_squad_cost
        elif use_chip == Chip.FREE_HIT:
          # Saves the current squad to be resetted later
          pre_free_hit_squad = current_squad.copy()

          # Using free hit does not affect budget
          free_hit_budget = current_budget + current_squad_cost
          current_squad = self.use_free_hit(gw_data, free_hit_budget)
        else:
          # Suggest and perform transfers
          suggested_transfers = self.team.suggest_transfers(gw_data, current_squad, current_budget)

          transfers = suggested_transfers[:transfers_available]
          current_squad, current_budget = self.transfer_players(current_squad, transfers, current_budget)
          transfers_available -= len(transfers)

          for transfer in transfers:
            print(f"Transfer: {transfer['out']['name']} => {transfer['in']['name']}")
          print("")

      print(f"GW{current_gw} Squad: {self._humanize_team_logs(current_squad)}")
      print(f"Budget: {current_budget}")
      print(f"Transfers Available: {transfers_available}\n")

      # Select best team and captain
      selected_team = self.team.pick_team(current_squad, gw_data)

      print(f"Selected team: {self._humanize_team_logs(selected_team)}")
      print(f"Selected Captain: {selected_team['captain']['name'].values[0]}")
      
      # Captaining, using triple captain if necessary
      triple_captain = False
      if use_chip == Chip.TRIPLE_CAPTAIN:
        self.use_triple_captain()

      if use_chip == Chip.BENCH_BOOST:
        self.use_bench_boost()

      gw_points = self.team.calc_team_points(
        gw_data,
        selected_team, 
        triple_captain=(use_chip == Chip.TRIPLE_CAPTAIN),
        bench_boost=(use_chip == Chip.BENCH_BOOST))

      total_points += gw_points
      
      print(f"Team Points: {gw_points}\n")

      if use_chip == Chip.FREE_HIT:
        current_squad = pre_free_hit_squad.copy()

      if self.show_optimal:
        # Compare to optimal
        optimal_squad, _, = self.team.get_best_squad(gw_data, self.MAX_BUDGET, optimal=True)
        optimal_team, optimal_captain = self.team.pick_team(optimal_squad, gw_data, optimal=True)
        optimal_points = self.team.calc_team_points(gw_data, optimal_team, optimal_captain)

        print(f"Optimal team: {self._humanize_team_logs(optimal_team)}")
        print(f"Optimal Captain: {optimal_captain['name']}")
        print(f"Optimal Points: {optimal_points}\n")
      
      print(f"Total points: {total_points}\n")
      print("---------------------------")
    
    print(f"Total profit: {self.total_profit}")
    print(f"Total loss: {self.total_loss}")

    print(f"\nTotal points: {total_points}\n")

  def _add_player_to_team(self, player, current_squad):
    player_position = player['position']
    player_df = pd.DataFrame([player])

    team_position = current_squad[player_position]
    current_squad[player_position] = pd.concat([team_position, player_df], ignore_index=True)

  def _remove_player_from_team(self, player, current_squad):
    player_id = player['id']
    player_position = player['position']

    team_position = current_squad[player_position]
    current_squad[player_position] = team_position[team_position['id'] != player_id]

  def transfer_players(self, current_squad, transfers, current_budget):
    for transfer in transfers:
      player_out = transfer['out']
      player_in = transfer['in']

      pos_players = current_squad[player_out['position']]
      player_then = pos_players[pos_players['id'] == player_out['id']].iloc[0]

      self._remove_player_from_team(player_out, current_squad)
      refund = self._calc_transfer_refund(player_then, player_out)
      current_budget += refund
      
      self._add_player_to_team(player_in, current_squad)
      current_budget -= player_in['cost']

    return current_squad, current_budget

  # Returns how much money should be added to budget when transfering out a player
  # Adheres to FPL profit and loss rules on transfers
  def _calc_transfer_refund(self, player_then, player_now):
    cost_then = player_then['cost']
    cost_now = player_now['cost']

    if cost_now > cost_then:
      profit = (cost_now - cost_then) / 2
      profit = math.floor(profit * 10) / 10

      self.total_profit += profit
      return cost_then + profit

    self.total_loss += cost_then - cost_now
    return cost_now

  # Check if any chip should be used
  # Returns the chip to be used or None
  def should_use_chip(self, current_squad, budget_available, gw_data, current_gw):
    # Check if should use triple captain
    if self.should_triple_captain(current_gw):
      return Chip.TRIPLE_CAPTAIN

    # Check if should use free hit
    if self.should_free_hit(current_squad, budget_available, gw_data, current_gw):
      return Chip.FREE_HIT

    # Check if should use wildcards
    if self.should_wildcard(current_squad, budget_available, gw_data, current_gw):
      return Chip.WILDCARD

    # Check if should use bench boost
    if self.should_bench_boost(current_gw):
      return Chip.BENCH_BOOST

    return None

  def should_triple_captain(self, current_gw):
    chip = Chip.TRIPLE_CAPTAIN
    if not self._is_chip_available(chip):
      return False

    if self._is_double_gameweek(current_gw):
      self.chips_available[chip] = False
      return True

    return False

  def use_triple_captain(self):
    self.chips_available[Chip.TRIPLE_CAPTAIN.value] = False

  def should_wildcard(self, current_squad, budget_available, gw_data, current_gw):
    """
    Determines whether to use the wildcard by checking if the current squad 
    significantly underperforms the best available squad.
    """
    chip = Chip.WILDCARD
    if (not self._is_chip_available(chip)) or current_gw == 1:
      return False

    # Key wildcard decision points
    first_quarter = self.MAX_GW * 1/4
    third_quarter = self.MAX_GW * 3/4

    strategy = self.chip_strategy[chip]

    # Only consider wildcarding at key points or if strategy is 'asap'
    should_first_half = current_gw >= first_quarter and current_gw < 20
    should_second_quarter = current_gw >= third_quarter and current_gw >= 20
    if strategy != 'asap' and not (should_first_half or should_second_quarter):
      return False

    # Get cached best squad
    cached_best_squad, _ = self._get_best_cached_squad(gw_data, budget_available, current_gw)

    squad_points = self.team.calc_squad_fitness(gw_data, current_squad)
    best_points = self.team.calc_squad_fitness(gw_data, self.cached_best_squad)

    return (best_points - squad_points) >= self.REVAMP_THRESHOLD

  def use_wildcard(self):
    if self.cached_best_squad is None:
      raise ValueError("Wildcard should not be used without calling should_wildcard first.")
        
    self.chips_available[Chip.WILDCARD.value] = False
    return self.cached_best_squad, self.cached_best_squad_cost

  def should_free_hit(self, current_squad, budget_available, gw_data, current_gw):
    chip = Chip.FREE_HIT
    if not self._is_chip_available(chip) or current_gw == 1:
      return False

    _, cached_best_squad_cost = self._get_best_cached_squad(gw_data, budget_available, current_gw)

    should_use = False

    strategy = self.chip_strategy[chip]
    if strategy == 'double_gw' and self._is_double_gameweek(current_gw):
      self.chips_available[chip] = False
      should_use = True
    elif strategy == 'blank_gw' and self._is_blank_gameweek(current_gw):
      if self._is_squad_affected_by_blank_gw(current_squad, gw_data):
        self.chips_available[chip] = False
        should_use = True

    squad_points = self.team.calc_squad_fitness(gw_data, current_squad)

    # Only use wildcard if it improves on current result
    if should_use and (cached_best_squad_cost - squad_points) >= self.REVAMP_THRESHOLD:
      self.chips_available[chip] = False
      return True

    return False

  def use_free_hit(self, gw_data, current_budget):
    if self.cached_best_squad is None:
      raise ValueError("Free Hit should not be used without calling should_free_hit first.")
        
    self.chips_available[Chip.FREE_HIT.value] = False
    return self.cached_best_squad

  def _is_squad_affected_by_blank_gw(self, current_squad, gw_data):
    # Get all teams that have fixtures in this gameweek
    active_teams = set(gw_data['team'].unique())

    # Get all teams from the user's squad
    squad_teams = set()

    for pos in self.POSITIONS:
      if not current_squad[pos].empty and 'team' in current_squad[pos]:
        squad_teams.update(current_squad[pos]['team'].unique())

    # Check if any team in the squad is missing from the active teams
    missing_teams = squad_teams - active_teams
    return len(missing_teams) > 0

  def should_bench_boost(self, current_gw):
    chip = Chip.BENCH_BOOST
    if not self._is_chip_available(chip):
      return False

    if self._is_double_gameweek(current_gw):
      self.chips_available[chip] = False
      return True

    return False

  def use_bench_boost(self):
    self.chips_available[Chip.BENCH_BOOST.value] = False

  def _get_best_cached_squad(self, gw_data, budget_available, current_gw):
    # Cache computed best squad
    if self.cached_best_squad is None or self.cached_best_squad_gw != current_gw:
      self.cached_best_squad, self.cached_best_squad_cost = self.team.get_best_squad(gw_data, budget_available)
      self.cached_best_squad_gw = current_gw

    return self.cached_best_squad, self.cached_best_squad_cost
  
  # Checks if current GW is double week
  def _is_double_gameweek(self, current_gw):
    gw_fixtures = self.fixtures_data[self.fixtures_data['GW'] == current_gw]

    # Count occurrences of each team in 'team_a' and 'team_h'
    team_counts = gw_fixtures[['team_a', 'team_h']].stack().value_counts()

    return any(team_counts > 1)

  # Checks if current GW is blank week
  def _is_blank_gameweek(self, current_gw):
    all_teams = set(range(1, 21))
    gw_fixtures = self.fixtures_data[self.fixtures_data['GW'] == current_gw]

    # Get all teams that have a fixture in this gameweek
    teams_with_fixtures = set(gw_fixtures[['team_a', 'team_h']].stack().unique())

    # Check if any team is missing (blank gameweek)
    return len(all_teams - teams_with_fixtures) > 0

  def _is_chip_available(self, chip):
    return self.chips_available.get(chip.value, False)

  def _load_predicted_gw_data(self, gw):
    filepath = f"predictions/gws/{self.season}/GW{gw}.csv"
    df = pd.read_csv(filepath)
    return df

  def _humanize_team_logs(self, selected_team):
    team_ids = { key: pd.DataFrame(selected_team[key])['name'].tolist() for key in selected_team }
    return team_ids

if __name__=='__main__':
  # Valid strategy choices
  valid_strategies = ["double_gw", "blank_gw"]
  valid_wildcard_strategies = ["asap", "wait"]
  valid_free_hit_strategies = ["double_gw", "blank_gw"]

  parser = argparse.ArgumentParser(description="Run the model with optional chip strategies.")

  parser.add_argument(
    "--season", type=str, nargs="?", default="2024-25",
    help="Season to simulate in the format 20xx-yy."
  )

  # Chip strategy arguments with validation
  # TODO: Update strategies
  parser.add_argument(
    "--triple_captain", type=str, choices=valid_strategies, default="double_gw",
    help="Strategy for the Triple Captain chip. Options: 'double_gw', 'blank_gw'."
  )
  parser.add_argument(
    "--wildcard", type=str, choices=valid_wildcard_strategies, default="double_gw",
    help="Strategy for the Wildcard chip. Options: 'double_gw', 'blank_gw', 'use_asap', 'wait'."
  )
  parser.add_argument(
    "--free_hit", type=str, choices=valid_free_hit_strategies, default="blank_gw",
    help="Strategy for the Free Hit chip. Options: 'double_gw', 'blank_gw'."
  )
  # TODO: Update strategies
  parser.add_argument(
    "--bench_boost", type=str, choices=valid_strategies, default="double_gw",
    help="Strategy for the Bench Boost chip. Options: 'double_gw', 'blank_gw'."
  )

  # Config parameters
  parser.add_argument(
    "--max_gw", type=int, nargs="?", default=38,
    help="Max GW to iterate through"
  )

  args = parser.parse_args()
  
  chip_strategy = {
    Chip.TRIPLE_CAPTAIN: args.triple_captain,
    Chip.WILDCARD: args.wildcard,
    Chip.FREE_HIT: args.free_hit,
    Chip.BENCH_BOOST: args.bench_boost
  }

  config = {
    'max_gw': args.max_gw
  }

  simulation = Simulation(
    season=args.season, 
    chip_strategy=chip_strategy,
    config=config)
  simulation.simulate_season()