import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import pandas as pd
from utils.model_types import ModelType
from utils.chips import Chip
from data.data_loader import DataLoader
from simulation import Simulation
import matplotlib.pyplot as plt

from evaluation.simulation.transfers_eval import TransfersEvaluation

HISTORY_TYPES = ['transfers', 'points', 'chips', 'diversity', 'budget']

class SimulationEvaluation:
  def __init__(
    self, 
    season,
    model=None, 
    chip_strategy=None,
    transfers_strategy=None,
    iterations=50,
    evaluations=[]
  ):
    self.season = season
    self.model = model
    self.chip_strategy = chip_strategy
    self.transfers_strategy = transfers_strategy
    self.evaluations = evaluations

    self.gw_data = self.data_loader.get_gw_predictions(season)

    self.ITERATIONS = iterations

  def _run_simulation(self):
    simulation = Simulation(
      season=self.season,
      model=self.model,
      chip_strategy=self.chip_strategy,
      transfers_strategy=self.transfers_strategy,
      debug=True
    )
    
    total_points, sim_histories = simulation.simulate_season()
    return total_points, sim_histories

  def evaluate(self):
    histories = { key: [] for key in HISTORY_TYPES }

    best_sim_points = 0
    best_sim_histories = None
    
    for i in range(self.ITERATIONS):
      print(f"Running simulation {i + 1}\n")

      sim_points, sim_histories = self._run_simulation()
      print("--------")

      for key in HISTORY_TYPES:
        histories[key].append(sim_histories[key])

      if best_sim_points < sim_points:
        best_sim_points = sim_points
        best_sim_histories = sim_histories

    if 'transfers' in self.evaluations:
      evaluation = self._evaluate_transfers(
        histories['transfers'], 
        best_sim_histories['transfers']
      )

    if 'points' in self.evaluations:
      evaluation = self._evaluate_points(
        histories['points'], 
        best_sim_histories['points']
      )

    if 'chips' in self.evaluations:
      evaluation = self._evaluate_chips(
        histories['chips'], 
        best_sim_histories['chips']
      )

    if 'diversity' in self.evaluations:
      evaluation = self._evaluate_diversity(
        histories['diversity'], 
        best_sim_histories['diversity']
      )

    if 'budget' in self.evaluations:
      evaluation = self._evaluate_budget(
        histories['budget'], 
        best_sim_histories['budget']
      )

  def _evaluate_transfers(self, histories, best_histories):
    transfers_eval = TransfersEvaluation(self.gw_data, self.transfers_strategy)
    return transfers_eval.evaluate(histories, best_histories)

  def _evaluate_points(self):
    print('here')
  
  def _evaluate_chips(self):
    print('here')
  
  def _evaluate_diversity(self):
    print('here')

  def _evaluate_budget(self):
    print('here')

  def _get_gw_data(self, gw):
    return self.gw_data[self.gw_data['GW'] == gw]

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run the model with optional chip strategies.")

  parser.add_argument(
    "--season", type=str, nargs="?", default="2023-24",
    help="Season to simulate in the format 20xx-yy."
  )

  parser.add_argument(
    '--model', type=str, help='The model to use', 
    choices=[m.value for m in ModelType]
  )

  parser.add_argument(
    '--iterations', type=int, help='Override number of iterations', 
    default=50
  )

  # Chip strategy arguments with validation
  valid_triple_captain_strategies = ["risky", "conservative"]
  parser.add_argument(
    "--triple_captain", type=str, choices=valid_triple_captain_strategies, default="conservative",
    help="Strategy for the Triple Captain chip. Options: 'risky', 'conservative'."
  )
  
  # TODO: Maybe include double and blank gws?
  valid_wildcard_strategies = ["asap", "wait"]
  parser.add_argument(
    "--wildcard", type=str, choices=valid_wildcard_strategies, default="wait",
    help="Strategy for the Wildcard chip. Options: 'asap', 'wait'."
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

  transfers_strategies=['simple', 'weighted']
  parser.add_argument(
    "--transfers", type=str, choices=transfers_strategies, default='simple',
    help="Strategy to calculate the fitness of transfer candidates. Options: 'simple', 'weighted'"
  )

  parser.add_argument(
    "--types", type=str, choices=HISTORY_TYPES,
    nargs="+", default=[], help="List of history types to evaluate"
  )

  args = parser.parse_args()

  if args.model:
    try:
      model_type = ModelType(args.model)
    except ValueError:
      print(f"Error: Invalid model type '{args.model}'. Choose from {', '.join(m.value for m in ModelType)}")
      exit(1)

  chip_strategy = {
    Chip.TRIPLE_CAPTAIN: args.triple_captain,
    Chip.WILDCARD: args.wildcard,
    Chip.FREE_HIT: args.free_hit,
    Chip.BENCH_BOOST: args.bench_boost
  }

  simulation_evaluation = SimulationEvaluation(
    season=args.season,
    model=args.model,
    chip_strategy=chip_strategy,
    transfers_strategy=args.transfers,
    iterations=args.iterations
  )
  simulation_evaluation.evaluate()