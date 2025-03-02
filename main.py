
from data.data_loader import DataLoader
from model import Model
from evaluate import Evaluate
from utils.model_types import ModelType
from models.lstm_model import LSTMModel
import argparse
from team import Team
from simulation import Simulation
from data.fbref.data_loader import DataLoader as FBref

def train_model(model, steps, season):
  # Use previous season as training
  start_year, end_year = season.split('-')
  new_start = int(start_year) - 1
  new_end = int(end_year) - 1
  prev_season = f"{new_start}-{new_end}"

  data_loader = DataLoader()

  training_data = data_loader.get_merged_gw_data(prev_season, steps)
  model.train(training_data)

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Run the model with optional training.')
  parser.add_argument('--train', action='store_true', help='Set this flag to true to train the model. Defaults to false.')
  parser.add_argument('--steps', type=int, nargs='?', const=5, default=5, help='Time step for data window. Defaults to 7 if not provided or null.')
  parser.add_argument('--season', type=str, nargs='?', default='2024-25', help='Season to simulate in the format 20xx-yy')
  
  # # Option to run predictions, true by default
  # parser.add_argument('--predict', action='store_true', help='Enable prediction using the trained model. Defaults to true.')
  # parser.add_argument('--no-predict', dest='predict', action='store_false', help='Disable prediction.')
  # parser.set_defaults(predict=True)

  args = parser.parse_args()

  data_loader = DataLoader()

  fixtures_data = data_loader.get_fixtures(args.season)
  teams_data = data_loader.get_teams_data(args.season)
  players_data = data_loader.get_players_data(args.season)
  gw_data = data_loader.get_merged_gw_data(args.season, args.steps)

  model = LSTMModel(
    gw_data=gw_data,
    teams_data=teams_data,
    fixtures=fixtures_data,
    players_data=players_data,
    season=args.season,
    time_steps=args.steps,
    train=args.train)

  if args.train:
    train_model(model, args.steps, args.season)

  model.predict_season()
  
#   evaluate = Evaluate(ids_df)
#   evaluate.evaulate_prediction(indexes, predictions, targets)
