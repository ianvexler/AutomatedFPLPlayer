import pandas as pd
import io
import soccerdata as sd
import requests
import json
import difflib

FILENAME = 'team_mapping.json'

class TeamMapping:
  def __init__(self):
    self.team_dict = {}

  def get_fpl_teams(self):
    # Fetch FPL team names
    response = requests.get("https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/master_team_list.csv")
    response.raise_for_status()
    team_df = pd.read_csv(io.StringIO(response.text))
    fpl_teams = team_df['team_name'].unique()
    return fpl_teams

  def get_club_elo_teams(self):
    # Fetch Club Elo team names
    club_elo = sd.ClubElo()
    current_elo = club_elo.read_by_date()
    current_elo = current_elo.reset_index()
    club_elo_teams = current_elo['team'].unique()
    return club_elo_teams

  def find_closest_match(self, team_name, team_list):
    # Find the closest match for a team name in a list of team names
    closest_matches = difflib.get_close_matches(team_name, team_list, n=1, cutoff=0.6)
    return closest_matches[0] if closest_matches else None

  def create_team_dict(self):
    # Fetch teams from FPL and Club Elo
    fpl_teams = self.get_fpl_teams()
    club_elo_teams = self.get_club_elo_teams()

    # Match FPL teams to closest Club Elo teams
    for fpl_team in fpl_teams:
      closest_club_elo_team = self.find_closest_match(fpl_team, club_elo_teams)
      self.team_dict[fpl_team] = {
        'FPL': fpl_team,
        'ClubElo': closest_club_elo_team
      }

  def get_team(self, team_name, source):
    # Retrieve the mapping for a specific team and source
    team_data = self.team_dict.get(team_name)
    if team_data:
      return team_data.get(source, f"No data for source {source}")
    else:
      return f"Team {team_name} not found in the dictionary."

  def save_dict_to_json(self):
    # Save the dictionary to a JSON file
    with open(FILENAME, 'w') as file:
      json.dump(self.team_dict, file, indent=4)

  def load_dict_from_json(self):
    # Load the dictionary from a JSON file
    with open(FILENAME, 'r') as file:
      self.team_dict = json.load(file)

# Run script from console
if __name__ == "__main__":
  team_mapping = TeamMapping()
  team_mapping.create_team_dict()
  team_mapping.save_dict_to_json()