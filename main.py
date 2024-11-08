
# from team_selector import TeamSelector
from data.data_loader import DataLoader
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def format_results(indexes, predictions, targets):
  combined_results = pd.DataFrame({
    'id': indexes,
    'test_total_points': predictions,
    'total_points': targets
  }).set_index('id')

  combined_results['difference'] = (combined_results['test_total_points'] - combined_results['total_points']).abs()

  # Sort by the difference
  sorted_combined_results = combined_results.sort_values(by='difference')
  sorted_combined_results.sort_index(inplace=True)

  return sorted_combined_results

if __name__=='__main__':
  player_id = '283'
  player_name = 'Mohamed Salah'
  gw_id = '1'

  data_loader = DataLoader()
  data = data_loader.get_data('2022-23')\
  
  x_train, x_test, y_train, y_test = data_loader.get_train_test_data(data)

  model = RandomForestRegressor()

  model.fit(x_train, y_train)

  predictions = model.predict(x_test)

  score = model.score(x_test, y_test)
  mse = mean_squared_error(y_test, predictions)

  print(f'R^2 score: {score}')
  print(f'Mean Squared Error: {mse}')

  test_indexes = x_test.index

  results_df = format_results(test_indexes, predictions, y_test)

  results_csv = results_df.to_csv('results.csv')




  # labels, results = data.get_data('2022-23')
  
  # labels_index = np.where(labels == 'total_points')[0][0]

  # model = RandomForestRegressor().fit()

  # print(results[0:, labels_index])

  # team_selector = TeamSelector(data)
  # best_team = team_selector.get_best_team()
  # print(best_team)