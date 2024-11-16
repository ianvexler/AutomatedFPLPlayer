import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from data.data_loader import DataLoader

class Evaluate:
  def __init__(self, ids_df: pd.DataFrame):
    self.ids_df = ids_df

  def evaulate_prediction(self, indexes, predictions, targets):
    error, mse, accuracy = self.score_model(predictions, targets)
    print(f'Mean Absolute Error: {error}')
    print(f'Mean Squared Error: {mse}')

    results = self.format_results(indexes, predictions, targets)
    results.to_csv('results.csv')

  """
  Calculate the error and mean squared error (MSE).

  Params:
      predictions: The predicted values.
      labels: The actual values.

  Returns:
    error: mean absolute error
    mse: mean squared error
  """
  def score_model(self, predictions, targets):
    predictions = predictions.round()

    error = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)

    return error, mse

  """
  Formats the results. Maps the FPL Ids to the player names
  """
  def format_results(self, indexes, predictions, targets):
    combined_results = pd.DataFrame({
      'id': indexes,
      'total_points': targets,
      'predicted_points': predictions,
    }).set_index('id')

    combined_results['difference'] = (combined_results['total_points'] - combined_results['predicted_points'])

    # Load the DataFrame containing player IDs and names, ensuring only FPL-related columns are selected
    fpl_ids_df = self.ids_df[['FPL_ID', 'FPL_Name']]

    # Merge combined_results with fpl_ids_df to include player names based on their IDs
    merged_results = combined_results.merge(fpl_ids_df, how='left', left_index=True, right_on='FPL_ID')

    # Drop 'FPL_ID' after merge since it's already the index
    merged_results.drop(columns=['FPL_ID'], inplace=True)
    
    # Sort by the difference
    sorted_combined_results = merged_results.sort_values(by='difference')
    # sorted_combined_results.sort_index(inplace=True)

    return sorted_combined_results