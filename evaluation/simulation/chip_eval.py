import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import pandas as pd
from utils.chips import Chip
from data.data_loader import DataLoader
from simulation import Simulation
import matplotlib.pyplot as plt
import seaborn as sns
import os

import json
from pathlib import Path
import pickle


class ChipEvaluation:
  def __init__(self, season, chip, target='xP', iterations=50, load_saved=False):
    self.season = season
    self.chip = chip
    self.target = target
    self.ITERATIONS = iterations
    self.load_saved = load_saved

    self.data_loader = DataLoader()
    self.gw_data = self.data_loader.get_gw_predictions(season)

    self.strategies = ['simple', 'weighted']
    self.DIRECTORY = f'plots/{chip.value}'
    os.makedirs(self.DIRECTORY, exist_ok=True)

  def _run_simulation(self, strategy):
    chip_strategy = {
      Chip.TRIPLE_CAPTAIN: 'conservative',
      Chip.WILDCARD: 'wait',
      Chip.FREE_HIT: 'blank_gw',
      Chip.BENCH_BOOST: "double_gw"
    }

    chip_strategy[self.chip] = strategy

    simulation = Simulation(
      season=self.season,
      transfers_strategy='weighted',
      chip_strategy=chip_strategy,
      debug=False
    )
    total_points, sim_histories = simulation.simulate_season()
    return total_points, sim_histories

  def _get_chip_strategy(self):
    match self.chip.value:
      case Chip.TRIPLE_CAPTAIN.value:
        return ["risky", "conservative"]
      case Chip.WILDCARD.value:
        return ["asap", "wait", "double_gw", "blank_gw"]
      case Chip.FREE_HIT.value:
        return ["double_gw", "blank_gw"]
      case Chip.BENCH_BOOST.value:
        return ["double_gw", "with_wildcard"]
    
    raise Exception('Invalid Chip')
      

  def evaluate(self):
    all_results = {}
    result_dir = Path(f"data/cached/chips/{self.chip}/{self.season}")
    result_dir.mkdir(parents=True, exist_ok=True)

    strategies = self._get_chip_strategy()

    if self.load_saved:
      print("Loading saved evaluation results...")
      for strategy in strategies:
        with open(result_dir / f"{strategy}_evaluation.json", "r") as f:
          all_results[strategy] = json.load(f)
    else:
      for strategy in strategies:
        print(f"\nEvaluating strategy: {strategy.upper()}\n")
        chip_histories = []
        points_histories = []

        best_sim_points = 0
        best_sim_chip_histories = None
        best_sim_gw_points = None

        for i in range(self.ITERATIONS):
            print(f"Running simulation {i + 1}")
            sim_points, sim_histories = self._run_simulation(strategy)

            sim_chips = sim_histories['chips']
            sim_points_per_gw = sim_histories['points']

            chip_histories.append(sim_chips)
            points_histories.append(sim_points_per_gw)

            if best_sim_points < sim_points:
              best_sim_points = sim_points
              best_sim_chip_histories = sim_chips
              best_sim_gw_points = sim_points_per_gw

        with open(result_dir / f"{strategy}_evaluation_metrics.pkl", "wb") as f:
          evaluation_metrics = {
            'chip_histories': chip_histories,
            'best_sim_chip_histories': best_sim_chip_histories,
            'best_sim_gw_points': best_sim_gw_points,
            'strategy': strategy,
            'points_histories': points_histories
          }
          pickle.dump(evaluation_metrics, f)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--season", type=str, default="2023-24")
  parser.add_argument("--target", type=str, default='xP', choices=['fpl_xP', 'xP'])
  parser.add_argument("--iterations", type=int, default=50)
  parser.add_argument("--load_saved", action="store_true", help="Load saved evaluation results instead of simulating")
  
  parser.add_argument("--chip", type=str, choices=[c.value for c in Chip])

  args = parser.parse_args()

  try:
    chip = Chip(args.chip)
  except ValueError:
    print(f"Error: Invalid model type '{args.model}'. Choose from {', '.join(m.value for m in ModelType)}")
    exit(1)

  evaluator = ChipEvaluation(
    season=args.season,
    target=args.target,
    iterations=args.iterations,
    load_saved=args.load_saved,
    chip=chip
  )
  evaluator.evaluate()
