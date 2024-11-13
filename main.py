
# from team_selector import TeamSelector
from data.data_loader import DataLoader
from data.fbref.data_loader import DataLoader as FBref

from model import Model

if __name__=='__main__':
  # data_loader = DataLoader()
  # data = data_loader.get_data('2022-23')\
  

  # model = Model(data)
  # results = model.predict()
  # results.to_csv('results.csv')


  fbref = FBref('2022-23')
  result = fbref.get_player_season_stats()








  # player_id = '283'
  # player_name = 'Mohamed Salah'
  # gw_id = '1'

  # labels, results = data.get_data('2022-23')
  
  # labels_index = np.where(labels == 'total_points')[0][0]

  # model = RandomForestRegressor().fit()

  # print(results[0:, labels_index])

  # team_selector = TeamSelector(data)
  # best_team = team_selector.get_best_team()
  # print(best_team)