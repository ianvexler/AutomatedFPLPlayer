
from data.data_loader import DataLoader
from model import Model
from evaluate import Evaluate
from utils.model_types import ModelType
from models.lstm_model import LSTMModel
import argparse
from team import Team
from simulation import Simulation

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Run the model with optional training.')
  parser.add_argument('--train', action='store_true', help='Set this flag to true to train the model. Defaults to false.')
  parser.add_argument('--steps', type=int, nargs='?', const=7, default=7, help='Time step for data window. Defaults to 7 if not provided or null.')
  parser.add_argument('--season', type=str, nargs='?', default='2024-25', help='Season to simulate in the format 20xx-yy')

  args = parser.parse_args()

  data_loader = DataLoader()

  fixtures_data = data_loader.get_fixtures(args.season)
  teams_data = data_loader.get_teams_data(args.season)

  # Uses data from the previous season
  players_data = data_loader.get_players_data(args.season)

  gw_data = data_loader.get_merged_gw_data(args.season, args.steps)

  model = LSTMModel(
    gw_data=gw_data,
    teams_data=teams_data,
    fixtures=fixtures_data,
    players_data=players_data,
    season=args.season,
    train=args.train,
    time_steps=args.steps)
  model.predict_season()
  
#   evaluate = Evaluate(ids_df)
#   evaluate.evaulate_prediction(indexes, predictions, targets)
