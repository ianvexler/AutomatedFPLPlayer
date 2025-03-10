import pandas as pd
import io
import os
import soccerdata as sd
import requests
import json
import difflib
from fuzzywuzzy import process, fuzz
import warnings
import re

class TeamMatcher:
  def __init__(self):
    self.FILENAME = 'team_ids.json'
    self.team_dict = self.load_dict()

    self.FPL_OVERRIDE = {
      "Spurs": "Tottenham Hotspur",
      "Man Utd": "Manchester Utd",
      "Man City": "Manchester City"
    }

    self.FBREF_DETAILS_OVERRIDE = {
      "Wolves": "Wolverhampton Wanderers FC"
    }

  def get_fpl_teams(self, start_year=20):
    teams_df = pd.DataFrame()

    # Fetch FPL team data
    for start_year in range(start_year, 25):
      season = f"20{start_year:02d}-{(start_year + 1) % 100:02d}"
      response = requests.get(f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/{season}/teams.csv")
      response.raise_for_status()
      
      season_df = pd.read_csv(io.StringIO(response.text))
      season_df["season"] = season
      
      if teams_df.empty:
        teams_df = season_df.copy()
      else:
        teams_df = pd.concat([teams_df, season_df], ignore_index=True)

    teams_df = teams_df.rename(columns={
      "name": "name"
    })
    return teams_df

  def get_club_elo_teams(self):
    # Fetch Club Elo team names
    club_elo = sd.ClubElo()
    current_elo = club_elo.read_by_date()
    current_elo = current_elo.reset_index()
    club_elo_teams = current_elo['team'].unique()
    return club_elo_teams

  def get_fbref_teams(self):
    seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(2020, 2025)]

    # Fetch Club Elo team names
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=FutureWarning)
      fbref = sd.FBref(leagues="ENG-Premier League", seasons=seasons)
      teams_data = fbref.read_team_season_stats()
    return teams_data

  # Gets the ids for english clubs
  def get_fbref_details(self):
    url = "https://fbref.com/en/country/clubs/ENG/"
    dfs = pd.read_html(url, extract_links='all')  # Extracts hyperlinks as (text, link) tuples
    df = dfs[0]

    # Filter out irrelevant teams
    df = df[df[('Comps', None)].apply(lambda x: int(x[0]) if isinstance(x, tuple) else 0) > 10]

    # First column contains tuples (Club Name, URL)
    teams_data = df.iloc[:, 0]

    # Convert tuple values directly into two columns
    teams_df = pd.DataFrame(teams_data.tolist(), columns=['name', 'url'])

    # Extract team ids
    teams_df['id'] = teams_df['url'].str.extract(r"/squads/([a-fA-F0-9]+)")[0]
    teams_df['url_name'] = teams_df['url'].str.extract(r"/history/([^/]+)$")[0]
    teams_df = teams_df.drop(columns=['url'])

    return teams_df

  def find_closest_elo_match(self, team_name, team_list):
    # Find the closest match for a team name in a list of team names
    closest_matches = difflib.get_close_matches(team_name, team_list, n=1, cutoff=0.6)
    return closest_matches[0] if closest_matches else None

  def find_closest_fbref_match(self, team_name, fbref_teams, fbref_details):
    closest_matches = process.extractOne(team_name, fbref_teams, scorer=fuzz.partial_ratio, score_cutoff=70)
    
    if closest_matches:
      match = closest_matches[0]
    else:
      # Try alternative matching method
      closest_matches = process.extractOne(team_name, fbref_teams, scorer=fuzz.token_sort_ratio, score_cutoff=70)
    
      if not closest_matches:
        raise Exception(f'No match for {team_name}')
      else:
        match = closest_matches[0]

    
    details_name = self.FBREF_DETAILS_OVERRIDE.get(match, match)
    details_match = process.extractOne(details_name, fbref_details['name'].values, scorer=fuzz.token_sort_ratio, score_cutoff=40)[0]

    if not details_match:
      raise Exception(f'No match for {team_name}')
    
    team_details = fbref_details[fbref_details['name'] == details_match]

    team_full_name = team_details['name'].iloc[0]
    team_id = team_details['id'].iloc[0]
    team_url_name = team_details['url_name'].iloc[0]

    return {
      "id": team_id,
      "name": match,
      "full_name": team_full_name.replace(" FC", ""),
      "url_name": re.sub(r"-Stats-and-History$", "", team_url_name)
    }

  def create_team_dict(self):
    self.team_dict = {}

    # Fetch teams from FPL and Club Elo
    fpl_teams = self.get_fpl_teams()
    club_elo_teams = self.get_club_elo_teams()

    fbref_data = self.get_fbref_teams()
    fbref_teams = fbref_data.reset_index()["team"].unique()
    fbref_details = self.get_fbref_details()

    fpl_team_names = fpl_teams['name'].unique()

    # Match FPL teams to closest Club Elo teams
    for fpl_team_name in fpl_team_names:
      team_name = self.FPL_OVERRIDE.get(fpl_team_name, fpl_team_name)

      closest_club_elo_team = self.find_closest_elo_match(team_name, club_elo_teams)
      closest_fbref_team = self.find_closest_fbref_match(team_name, fbref_teams, fbref_details)

      fpl_team_entries = fpl_teams[fpl_teams["name"] == fpl_team_name]

      self.team_dict.setdefault(fpl_team_name, {
        'ClubElo': closest_club_elo_team,
        'FBref': closest_fbref_team,
        'FPL': {}
      })
      
      for _, fpl_team in fpl_team_entries.iterrows():
        season = fpl_team['season']

        self.team_dict[fpl_team_name]['FPL'][season] = {
          "id": fpl_team['id'], 
          "name": fpl_team['name']
        }

  def get_fpl_team(self, key, season, key_type='id'):
    for _, team_data in self.team_dict.items():
      fpl_data = team_data.get("FPL", {})

      # Check if the season exists in FPL data
      if season in fpl_data:
        team_entry = fpl_data[season]

        # Check based on key_type
        if key_type in team_entry and team_entry[key_type] == key:
          return team_data  

    raise Exception(f"Team could not be found for key {key} with type {key_type} in season {season}")

  def get_fbref_team(self, key, key_type='name'):
    for _, team_data in self.team_dict.items():
      team_entry = team_data.get("FBref", {})

      # Check based on key_type
      if key_type in team_entry and team_entry[key_type] == key:
        return team_data  

    raise Exception(f"Team could not be found for key {key} with type {key_type} in season {season}")

  """Returns a list of teams that have an FPL entry for the given season."""
  def get_season_teams(self, season):
    return {team: data for team, data in self.team_dict.items() if season in data.get("FPL", {})}

  def save_dict_to_json(self):
    # Save the dictionary to a JSON file
    with open(self.FILENAME, 'w') as file:
      json.dump(self.team_dict, file, indent=4)
    return True

  def load_dict(self):
    script_dir = os.path.dirname(os.path.abspath(__file__))    
    file_path = os.path.join(script_dir, self.FILENAME)

    with open(file_path, 'r') as file:
      self.team_dict = json.load(file)
    return self.team_dict

if __name__ == "__main__":
  team_matcher = TeamMatcher()
  team_matcher.create_team_dict()
  team_matcher.save_dict_to_json()