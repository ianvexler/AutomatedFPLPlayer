import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

class SimulationEvaluation:
  def __init__(
    self, 
    season,
    model=None, 
    chip_strategy=None,
    iterations=50
  ):
    self.season = season
    self.model = model
    self.chip_strategy = chip_strategy

    self.gw_data = self._load_gw_data()

    self.ITERATIONS = iterations

  def _run_simulation(self):
    simulation = Simulation(
      season=self.season,
      model=self..model,
      chip_strategy=self.chip_strategy,
      config=config,
      debug=False
    )
    
    total_points, sim_histories = simulation.simulate_season()
    return total_points, sim_histories

  def evaluate(self):
    history_types = ['transfers', 'points', 'chips', 'diversity', 'budget']
    histories = { key: [] for key in history_types }

    for _ in range(self.ITERATIONS):
      total_points, sim_histories = self._run_simulation()

      for key in history_types:
        histories[key].append(sim_histories[key])

    # Iterates through all history types and calls the corresponding evaluation method
    for method_name in history_types:
      evaluation_method = getattr(self, f"_evaluate_{method_name}")
      evaluation_method()

  def _evaluate_transfers(self):
    print('here')

  def _evaluate_points(self):
    print('here')
  
  def _evaluate_chips(self):
    print('here')
  
  def _evaluate_diversity(self):
    print('here')

  def _evaluate_budget(self):
    print('here')

  def _load_gw_data(self):
    # TODO: This has to be improved to match prediction file format
    # Also has to load all gws
    if self.model:
      filepath = f"predictions/{model}/gws/{self.season}/GW{gw}.csv"
      df = pd.read_csv(filepath)
      return df
    else:
      gw_data = self.data_loader.get_merged_gw_data(self.season)
      return gw_data

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run the model with optional chip strategies.")

  parser.add_argument(
    "--season", type=str, nargs="?", default="2024-25",
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
    iterations=args.iterations
  )
  simulation_evaluation.evaluate()