import pandas as pd
from pathlib import Path

class DataLoader:
  def __init__(self, season):
    self.season = season

  def get_gw_predictions(self):
    project_root = Path(__file__).resolve().parent
    directory = project_root / 'data' / 'gws' / self.season

    if directory.exists():
      predictions_df = pd.concat(
        [pd.read_csv(csv_file) for csv_file in directory.glob("*.csv")],
        ignore_index=True
      )
      return predictions_df
    else:
      raise Exception(f"Predictions not found in {directory}")
