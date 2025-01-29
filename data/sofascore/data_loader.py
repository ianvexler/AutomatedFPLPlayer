import soccerdata as sd
import pandas as pd

class DataLoader:
  def __init__(self, season):
    season = self.__convert_season_format(season)

    self.season = season
    self.sofascore = sd.Sofascore(leagues='ENG-Premier League', seasons=season)

  """
  Converts the season to the correct format (e.g. 2022-23 to 2022/2023)
  """
  def __convert_season_format(self, season) -> str:
    season_corrected = season.replace('-', '/')
    start_year, end_year = season_corrected.split('/')
    
    end_year_full = f"20{end_year}"
    
    return f"{start_year}/{end_year_full}"

  def get_schedule(self):
    raw_schedule = self.sofascore.read_schedule()

    schedule = self.sanitize_schedule(raw_schedule)
    return schedule

  def sanitize_schedule(self, data):
    data = data.rename(columns={"round": "GW"})
    data = data.reset_index()

    sanitized_data = data[["game", "GW"]]

    # Splits the data data into three columns
    game_data = sanitized_data["game"]

    sanitized_data.loc[:, ['date', 'home_team', 'away_team']] = sanitized_data['game'].apply(
      lambda game: pd.Series(self.split_game_string(game))
    )
    sanitized_data = sanitized_data.drop('game', axis=1)

    return sanitized_data

  """
  Splits a game string into three strings

  Params:
    - game_string: The string consisiting of f"{date} {home_team}-{away_team}" (e.g. 2022-08-05 Crystal Palace-Arsenal)

  Returns:
    - date: The date of the game (e.g. 2022-08-05)
    - home_team: The home team (e.g. Crystal Palace)
    - away_team: The away team (e.g. Arsenal)
  """
  def split_game_string(self, game_string):
    date, teams = game_string.split(' ', 1)    
    home_team, away_team = teams.split('-')
    
    return date, home_team, away_team

