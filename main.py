
from data.data_loader import DataLoader
from model import Model
from evaluate import Evaluate
from utils.model_types import ModelType
from models.lstm_model import LSTMModel
import argparse

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Run the model with optional training.')
  parser.add_argument('--train', action='store_true', help='Set this flag to true to train the model. Defaults to false.')
  args = parser.parse_args()

  data_loader = DataLoader()

  fixtures = data_loader.get_fixtures('2023-24')
  teams_data = data_loader.get_teams_data('2023-24')

  # Uses data from the previous season
  gw_data = data_loader.get_merged_gw_data('2022-23') 
  season_data = data_loader.get_data('2022-23')
  players_data = data_loader.get_players_data('2022-23')

  model = LSTMModel(
    season_data=season_data, 
    gw_data=gw_data,
    teams_data=teams_data,
    fixtures=fixtures,
    players_data=players_data,
    season='2023-24',
    train=args.train)
  
  model.predict_season()

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
