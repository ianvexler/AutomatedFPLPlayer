from data.vaastav.data_loader import DataLoader as Vaastav

class DataLoader:
  def get_data(self, season):
    data_loader = Vaastav(season)
    data = data_loader.get_full_season_data()

    return data
