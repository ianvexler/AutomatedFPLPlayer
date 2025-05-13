from utils.team_manager import TeamManager
from data.data_loader import DataLoader
import pandas as pd
import argparse
import math
from utils.model_types import ModelType
from utils.chips import Chip

DEFAULT_CONFIG = { 'max_gw': 38 }

class Simulation:
  def __init__(
    self, 
    season, 
    source=None, 
    chip_strategy=None, 
    show_optimal=False, 
    selection_strategy='weighted',
    config=DEFAULT_CONFIG,
    target='xP',
    debug=False
  ):
    self.season = season

    if not chip_strategy:
      # Default chip strategy
      chip_strategy = {
        Chip.TRIPLE_CAPTAIN: 'conservative',
        Chip.WILDCARD: 'wait',
        Chip.FREE_HIT: 'double_gw',
        Chip.BENCH_BOOST: "with_wildcard"
      }

    self.chip_strategy = chip_strategy
    self.selection_strategy = selection_strategy
    self.debug = debug

    self.POSITIONS = ['GK', 'DEF', 'MID', 'FWD']
    
    self.chips_available = { chip: True for chip in Chip }

    self.data_loader = DataLoader()
    self.fixtures_data = self.data_loader.get_fixtures(season)
    self.teams_data = self.data_loader.get_teams_data(season)
    self.gw_data = self.data_loader.get_gw_predictions(season)

    self.target = target

    self.team_manager = TeamManager(
      season=self.season,
      teams_data=self.teams_data,
      fixtures_data=self.fixtures_data,
      selection_strategy=self.selection_strategy,
      target=self.target
    )

    self.MAX_GW = config['max_gw']
    self.MAX_BUDGET = 1000
    self.MAX_FREE_TRANSFERS = 5 if season == '2024-25' else 2
    self.REVAMP_THRESHOLD = 40

    self.cached_best_squad = None
    self.cached_best_squad_cost = None
    self.cached_best_squad_gw = None

    self.total_profit = 0
    self.total_loss = 0

    self.show_optimal = show_optimal

    # History of metrics to evaluate
    self.transfer_history = {}
    self.point_history = {}
    self.chip_history = {
      Chip.TRIPLE_CAPTAIN: None,
      Chip.WILDCARD: [],
      Chip.FREE_HIT: None,
      Chip.BENCH_BOOST: None
    }
    self.diversity_history = {}
    self.budget_history = {}
    self.team_history = {}

  def simulate_season(self):
    """
    Simulates a season by iterating through every GW. This is the core of the season simulator.

    All strategical decisions are performed in this function. This includes:
    - Transfers
    - Chip usage
    """
    print(f"Simulating season {self.season}\n")

    # Ensure variables are restarted
    current_gw = 1
    current_budget = self.MAX_BUDGET
    transfers_available = 0

    total_points = 0

    current_squad = {}
    current_squad_cost = 0
    
    total_team_diversity = 0

    # Iterates through every GW, including GW1
    for current_gw in range(1, self.MAX_GW + 1):
      if current_gw > 1 and transfers_available < self.MAX_FREE_TRANSFERS:
        transfers_available += 1

      # Restart wildcard on GW20
      if current_gw == 20:
        self.chips_available[Chip.WILDCARD] = True

      # Load GW data
      gw_data = self._get_gw_data(current_gw)

      if current_gw == 1:
        # Set initial team for GW1
        current_squad, current_squad_cost = self.team_manager.get_best_squad(gw_data, current_budget, current_gw)
        current_budget -= current_squad_cost
      
      budget_available = current_budget + current_squad_cost
      use_chip = self.should_use_chip(current_squad, budget_available, gw_data, current_gw)
    
      if use_chip:
        if self.debug:
          print(f"Using chip: {use_chip.value.replace("_", " ").title()}\n")

        # Add budget to history
        if use_chip.value == Chip.WILDCARD.value:
          self.chip_history[use_chip].append(current_gw)
        else:
          self.chip_history[use_chip] = current_gw

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
          force_transfer = transfers_available == self.MAX_FREE_TRANSFERS
          suggested_transfers = self.team_manager.suggest_transfers(
            gw_data, 
            current_gw,
            current_squad, 
            current_budget, 
            force_transfer
          )

          current_squad, current_budget, transfers = self.transfer_players(
            current_squad, 
            suggested_transfers, 
            current_budget, 
            transfers_available
          )
          transfers_available -= len(transfers)

          # Add budget to history
          self.budget_history[current_gw] = current_budget

          self.transfer_history.setdefault(current_gw, [])

          for transfer in transfers:
            if self.debug:
              print(f"Transfer: {transfer['out']['name']} => {transfer['in']['name']}")

            # Add transfer to history
            self.transfer_history[current_gw].append(transfer)
          
      # Select best team and captain
      selected_team = self.team_manager.pick_team(current_squad, gw_data)
      self.team_history[current_gw] = selected_team

      current_team_diversity = self.team_manager.calc_team_diversity(selected_team)
      total_team_diversity += current_team_diversity

      # Add diversity to history
      self.diversity_history[current_gw] = current_team_diversity

      if self.debug:
        print(f"\nGW{current_gw} Squad: {self._humanize_team_logs(current_squad)}")
        print(f"Budget: {current_budget}")
        print(f"Transfers Available: {transfers_available}\n")

        print(f"Selected Team: {self._humanize_team_logs(selected_team)}")
        print(f"Selected Captain: {selected_team['captain']['name'].values[0]}")

        print(f"\nTeam Diversity: {current_team_diversity}")

      # Captaining, using triple captain if necessary
      triple_captain = False
      
      # TODO: Accept multiple tripple captain strats
      if use_chip == Chip.TRIPLE_CAPTAIN:
        self.use_triple_captain()

      if use_chip == Chip.BENCH_BOOST:
        self.use_bench_boost()

      gw_points = self.team_manager.calc_team_points(
        gw_data,
        selected_team, 
        triple_captain=(use_chip == Chip.TRIPLE_CAPTAIN),
        bench_boost=(use_chip == Chip.BENCH_BOOST),
        )
      
      self.point_history[current_gw] = gw_points

      total_points += gw_points

      if self.debug:
        print(f"Team Points: {gw_points}\n")

      if use_chip == Chip.FREE_HIT:
        current_squad = pre_free_hit_squad.copy()

      if self.show_optimal:
        # Compare to optimal
        optimal_squad, _, = self.team_manager.get_best_squad(gw_data, self.MAX_BUDGET, current_gw, optimal=True)
        
        if optimal_squad:
          optimal_team, optimal_captain = self.team_manager.pick_team(optimal_squad, gw_data, optimal=True)
          optimal_points = self.team_manager.calc_team_points(gw_data, optimal_team, optimal_captain)

          optimal_team_diversity = self.team_manager.calc_team_diversity(optimal_team)

          if self.debug:
            print(f"Optimal Team: {self._humanize_team_logs(optimal_team)}")
            print(f"Optimal Captain: {optimal_captain['name']}")
            print(f"Optimal Points: {optimal_points}\n")

            print(f"\nOptimal Team Diversity: {optimal_team_diversity}")
      
      if self.debug:
        print(f"Total Points: {total_points}\n")
        print("---------------------------")
    
    print(f"Total Profit: {self.total_profit}")
    print(f"Total Loss: {self.total_loss}")

    print(f"\nTotal Points: {total_points}")

    # Calculates the mean team diversity
    mean_team_diversity = total_team_diversity / self.MAX_GW
    print(f"\nMean Team Diversity: {mean_team_diversity}\n")

    evaluation_histories = self._build_eval_histories()
    return total_points, evaluation_histories

  def _build_eval_histories(self):
    """
    Builds a dict containing all the potentially useful histories for evaluation
    """
    return {
      'transfers': self.transfer_history,
      'points': self.point_history,
      'chips': self.chip_history,
      'diversity': self.diversity_history,
      'budget': self.budget_history,
      'teams': self.team_history
    }

  def _add_player_to_team(self, player, current_squad):
    """
    Adds a player to a squad
    """
    player_position = player['position']
    player_df = pd.DataFrame([player])

    team_position = current_squad[player_position]
    current_squad[player_position] = pd.concat([team_position, player_df], ignore_index=True)

  def _remove_player_from_team(self, player, current_squad):
    """
    Removes a player from a squad
    """
    player_id = player['id']
    player_position = player['position']

    team_position = current_squad[player_position]
    current_squad[player_position] = team_position[team_position['id'] != player_id]

  def transfer_players(self, current_squad, transfers, current_budget, transfers_available):
    """
    Executes a list of player transfers within the constraints of budget and available free transfers.

    For each proposed transfer:
    - Ensures it does not exceed the allowed number of transfers.
    - Checks that the transfer is affordable within the current budget.
    - Removes the outgoing player, refunds their value (considering buy/sell price logic), and adds the new player.

    Returns:
        tuple: Updated squad (dict), updated budget (float), and list of performed transfers.
    """
    transfer_count = 0

    performed_transfers = []

    # Iterates through every transfer
    for transfer in transfers:
      if transfer_count == transfers_available:
        break

      player_out = transfer['out']
      player_in = transfer['in']

      pos_players = current_squad[player_out['position']]
      player_then = pos_players[pos_players['id'] == player_out['id']].iloc[0]

      refund = self._calc_transfer_refund(player_then, player_out)
  
      # Ensures budget conditions are met
      if current_budget + refund - player_in['cost'] < 0:
        continue

      self._remove_player_from_team(player_out, current_squad)
      current_budget += refund
      
      self._add_player_to_team(player_in, current_squad)
      current_budget -= player_in['cost']

      performed_transfers.append(transfer)
      
      transfer_count += 1

    return current_squad, current_budget, performed_transfers

  def _calc_transfer_refund(self, player_then, player_now):
    """
    Returns how much money should be added to budget when transfering out a player
    Adheres to FPL profit and loss rules on transfers

    Returns:
      The cost to be refunded for a player
    """
    cost_then = player_then['cost']
    cost_now = player_now['cost']

    if cost_now > cost_then:
      profit = (cost_now - cost_then) / 2
      profit = math.floor(profit * 10) / 10

      self.total_profit += profit
      return cost_then + profit

    self.total_loss += cost_then - cost_now
    return cost_now

  
  def should_use_chip(self, current_squad, budget_available, gw_data, current_gw):
    """
    Check if any chip should be used
    
    Returns
      Returns the chip to be used or None
    """

    # Check if should use triple captain
    if self.should_wildcard(current_squad, budget_available, gw_data, current_gw):
      return Chip.WILDCARD

    # Check if should use bench boost (e.g. after wildcard)
    if self.should_bench_boost(current_gw):
      return Chip.BENCH_BOOST

    # Check if should use free hit
    if self.should_free_hit(current_squad, budget_available, gw_data, current_gw):
      return Chip.FREE_HIT

    # Check if should use triple captain
    if self.should_triple_captain(current_gw):
      return Chip.TRIPLE_CAPTAIN

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
    self.chips_available[Chip.TRIPLE_CAPTAIN] = False

  def should_wildcard(self, current_squad, budget_available, gw_data, current_gw):
    """
    Determines whether to use the wildcard chip based on the selected strategy.

    Returns:
      bool: True if wildcard should be used, otherwise False.
    """
    chip = Chip.WILDCARD
    if (not self._is_chip_available(chip)) or current_gw == 1:
      return False

    strategy = self.chip_strategy[chip]
    should_use = False

    if strategy == 'double_gw' and self._is_double_gameweek(current_gw):
      self.chips_available[chip] = False
      return True

    if strategy == 'asap' or strategy == 'wait':
      # Key wildcard decision points
      first_quarter = self.MAX_GW * 1/4
      third_quarter = self.MAX_GW * 3/4

      strategy = self.chip_strategy[chip]

      # Only consider wildcarding at key points or if strategy is 'asap'
      should_first_half = current_gw >= first_quarter and current_gw < 20
      should_second_quarter = current_gw >= third_quarter and current_gw >= 20
      if not(strategy != 'asap' and not (should_first_half or should_second_quarter)):
        should_use = True

    if should_use:
      # Get cached best squad
      cached_best_squad, _ = self._get_best_cached_squad(gw_data, budget_available, current_gw)

      squad_points = self.team_manager.calc_squad_fitness(gw_data, current_squad)
      best_points = self.team_manager.calc_squad_fitness(gw_data, self.cached_best_squad)
      
      return (best_points - squad_points) > 0
    
    return False

  def use_wildcard(self):
    if self.cached_best_squad is None:
      raise ValueError("Wildcard should not be used without calling should_wildcard first.")
        
    self.chips_available[Chip.WILDCARD] = False
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

    best_points = self.team_manager.calc_squad_fitness(gw_data, self.cached_best_squad)
    squad_points = self.team_manager.calc_squad_fitness(gw_data, current_squad)

    # Only use wildcard if it improves on current result
    if should_use and (best_points - squad_points) > 0:
      self.chips_available[chip] = False
      return True

    return False

  def use_free_hit(self, gw_data, current_budget):
    if self.cached_best_squad is None:
      raise ValueError("Free Hit should not be used without calling should_free_hit first.")
        
    self.chips_available[Chip.FREE_HIT] = False
    return self.cached_best_squad

  def _is_squad_affected_by_blank_gw(self, current_squad, gw_data):
    gw_data = gw_data.copy()
    
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

    strategy = self.chip_strategy[chip]

    if strategy == 'double_gw':
      if self._is_double_gameweek(current_gw):
        self.chips_available[chip] = False
        return True

    # Uses bench boost after wildcard has been played
    if strategy == 'with_wildcard':
      if not self.chips_available[Chip.WILDCARD]:
        self.chips_available[chip] = False
        return True

    return False

  def use_bench_boost(self):
    self.chips_available[Chip.BENCH_BOOST] = False

  def _get_best_cached_squad(self, gw_data, budget_available, current_gw):
    """
    Gets the best squad for a GW given a budget and saves it
    If squad has already been generated for a GW it returns the saved best 

    Returns
      The best cached squad for a GW
    """
    gw_data = gw_data.copy()

    # Cache computed best squad
    if self.cached_best_squad is None or self.cached_best_squad_gw != current_gw:
      best_squad, best_squad_cost = self.team_manager.get_best_squad(gw_data, budget_available, current_gw)
      
      if best_squad:
        self.cached_best_squad, self.cached_best_squad_cost = best_squad, best_squad_cost

      self.cached_best_squad_gw = current_gw

    return self.cached_best_squad, self.cached_best_squad_cost
  
  
  def _is_double_gameweek(self, current_gw):
    """
    Checks if current GW is double week
    """
    gw_fixtures = self.fixtures_data[self.fixtures_data['GW'] == current_gw]

    # Count occurrences of each team in 'team_a' and 'team_h'
    team_counts = gw_fixtures[['team_a', 'team_h']].stack().value_counts()

    return any(team_counts > 1)

  def _is_blank_gameweek(self, current_gw):
    """
    Checks if current GW is blank week
    """
    all_teams = set(range(1, 21))
    gw_fixtures = self.fixtures_data[self.fixtures_data['GW'] == current_gw]

    # Get all teams that have a fixture in this gameweek
    teams_with_fixtures = set(gw_fixtures[['team_a', 'team_h']].stack().unique())

    # Check if any team is missing (blank gameweek)
    return len(all_teams - teams_with_fixtures) > 0

  def _is_chip_available(self, chip):
    return self.chips_available.get(chip, False)

  def _get_gw_data(self, gw):
    return self.gw_data[self.gw_data['GW'] == gw]

  def _humanize_team_logs(self, selected_team):
    """
    Improves the logs for the terminal to visualize squad composition
    """
    team_ids = { key: pd.DataFrame(selected_team[key])['name'].tolist() for key in selected_team }
    return team_ids

if __name__=='__main__':
  # Valid strategy choices
  parser = argparse.ArgumentParser(description="Run the model with optional chip strategies.")

  parser.add_argument(
    "--season", type=str, nargs="?", default="2023-24",
    help="Season to simulate in the format 20xx-yy."
  )

  parser.add_argument(
    '--target', type=str, help='The expected points target to use', default='xP', 
    choices=['fpl_xP', 'xP']
  )

  # Chip strategy arguments with validation
  valid_triple_captain_strategies = ["risky", "conservative"]
  parser.add_argument(
    "--triple_captain", type=str, choices=valid_triple_captain_strategies, default="conservative",
    help="Strategy for the Triple Captain chip. Options: 'risky', 'conservative'."
  )
  
  valid_wildcard_strategies = ["asap", "wait", "double_gw"]
  parser.add_argument(
    "--wildcard", type=str, choices=valid_wildcard_strategies, default="wait",
    help="Strategy for the Wildcard chip. Options: 'asap', 'wait', 'double_gw'."
  )
  
  valid_free_hit_strategies = ["double_gw", "blank_gw"]
  parser.add_argument(
    "--free_hit", type=str, choices=valid_free_hit_strategies, default="blank_gw",
    help="Strategy for the Free Hit chip. Options: 'double_gw', 'blank_gw'."
  )
  
  valid_bench_boost_strategies = ["double_gw", "with_wildcard"]
  parser.add_argument(
    "--bench_boost", type=str, choices=valid_bench_boost_strategies, default="double_gw",
    help="Strategy for the Bench Boost chip. Options: 'double_gw', 'with_wildcard'."
  )

  selection_strategies=['simple', 'weighted']
  parser.add_argument(
    "--selection_strat", type=str, choices=selection_strategies, default='simple',
    help="Strategy to calculate the fitness of transfer candidates. Options: 'simple', 'weighted'"
  )

  # Config parameters
  parser.add_argument(
    "--max_gw", type=int, nargs="?", default=38,
    help="Max GW to iterate through. Only to debug"
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
    target=args.target,
    chip_strategy=chip_strategy,
    selection_strategy=args.selection_strat,
    config=config,
    debug=True
  )
  simulation.simulate_season()