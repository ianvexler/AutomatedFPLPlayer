
from data.data_loader import DataLoader
from model import Model
from evaluate import Evaluate
from utils.model_types import ModelType
from models.lstm_model import LSTMModel
import argparse
from team import Team

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Run the model with optional training.')
  parser.add_argument('--train', action='store_true', help='Set this flag to true to train the model. Defaults to false.')
  parser.add_argument('--steps', type=int, nargs='?', const=7, default=7, help='Time step for data window. Defaults to 7 if not provided or null.')
  
  args = parser.parse_args()

  data_loader = DataLoader()

  fixtures = data_loader.get_fixtures('2023-24')
  teams_data = data_loader.get_teams_data('2023-24')

  # Uses data from the previous season
  players_data = data_loader.get_players_data('2023-24')

  team_selector = Team(players_data)
  initial_team = team_selector.initial_team()
  print(initial_team)

  # gw_data = data_loader.get_merged_gw_data('2023-24', args.steps)

  # model = LSTMModel(
  #   gw_data=gw_data,
  #   teams_data=teams_data,
  #   fixtures=fixtures,
  #   players_data=players_data,
  #   season='2023-24',
  #   train=args.train,
  #   time_steps=args.steps)
  
  # model.predict_season()

# if __name__=='__main__':
#   data_loader = DataLoader()
#   train_data = data_loader.get_data('2022-23')
#   data = data_loader.get_data('2023-24')
  
#   model = Model(data, train_data, model=ModelType('xgboost'))
#   # indexes, predictions, targets = model.predict()
#   indexes, predictions, targets = model.test_predict()

#   ids_df = data_loader.get_id_dict_data('2022-23')[['FPL_ID', 'FPL_Name']]

#   evaluate = Evaluate(ids_df)
#   evaluate.evaulate_prediction(indexes, predictions, targets)
