import pandas as pd

class Evaluate:
  def format_results(self, indexes, predictions, targets):
    combined_results = pd.DataFrame({
      'id': indexes,
      'total_points': targets,
      'test_total_points': predictions,
    }).set_index('id')

    combined_results['difference'] = (combined_results['total_points'] - combined_results['test_total_points'])

    # Sort by the difference
    sorted_combined_results = combined_results.sort_values(by='difference')
    # sorted_combined_results.sort_index(inplace=True)

    return sorted_combined_results