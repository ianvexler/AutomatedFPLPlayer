import soccerdata as sd
import pandas as pd

class DataLoader():
  def __init__(self):
    self.club_elo = sd.ClubElo()

  def get_team_history(team_name):
    club_data = self.club_elo.read_team_history(team_name)
