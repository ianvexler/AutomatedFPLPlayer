import os
import sys
import difflib

# TODO: Make this a module
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import pandas as pd
import requests
import json
import io
from data.fbref.data_loader import DataLoader as FBref
from thefuzz import process, fuzz
import numpy as np
from utils.team_matcher import TeamMatcher

class PlayerMatcher:
  def __init__(self):
    self.FILENAME = 'player_ids.json'
    self.player_dict = self.load_dict_from_json()

    # TODO: Maybe include saves?
    self.COLUMNS = ["team", "name", "goals", "assists", "id"]
    # self.COLUMNS = ["season", "team", "name", "nation", "pos", "age", "goals", "assists"]

  def get_fpl_players(self, season):
    # Fetch FPL team names
    response = requests.get(f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/{season}/players_raw.csv")
    response.raise_for_status()
    players_df = pd.read_csv(io.StringIO(response.text))
    
    normalized_df = self._normalize_fpl_data(players_df)
    return normalized_df

  # All data should be normlaized for matching
  def _normalize_fpl_data(self, data):
    data = data.copy()

    # Combine first and second name
    data["name"] = data["first_name"] + " " + data["second_name"]
    data = data.drop(columns=["first_name", "second_name"])

    data = data.rename(columns={
      "goals_scored": "goals",
    })

    # Other columns to include
    stats_columns = ["team", "goals", "assists", "saves"]
    data[stats_columns] = data[stats_columns].apply(pd.to_numeric, errors="coerce")

    required_columns = ["web_name"] + self.COLUMNS
    data = data[required_columns]

    return data

  def get_fbref_data(self, season):
    fbref = FBref(season)
    players_df = fbref.get_player_season_stats()
    player_ids_df = fbref.get_players_ids()

    players_df = self._normalize_fbref_data(players_df, player_ids_df)
    return players_df

  def _normalize_fbref_data(self, data, ids):
    # Extract all Premier League data
    data = data.xs("ENG-Premier League", level="league").reset_index()

    # Flatten MultiIndex columns by joining tuples into strings (e.g., ('Performance', 'Gls') -> 'Gls')
    data.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]

    # Rename relevant columns for consistency
    data = data.rename(columns={
      "player": "name",
      "squad": "team",
      "Performance Gls": "goals", 
      "Performance Ast": "assists"
    })

    data = data.merge(
      ids,
      how='left',
      left_on='name',
      right_on='name'
    )

    # Select only relevant columns
    columns = ["position"] + self.COLUMNS
    data = data[columns]

    # Merge stats for players who played in multiple teams in a season
    data = data.groupby(["id", "name", "position"]).agg({
      "team": lambda x: ";".join(sorted(set(x))),  # Merge unique team names
      "goals": "sum",  # Sum goals across teams
      "assists": "sum"  # Sum assists across teams
    }).reset_index()

    return data
  
  def create_player_dict(self, start_year=17):
    # Restart player dict
    self.player_dict = {}

    # Iterates through every season starting from starting_year
    for start_year in range(start_year, 24):
      season = f"20{start_year:02d}-{(start_year + 1) % 100:02d}"
      
      fbref_data = self.get_fbref_data(season)
      fpl_data = self.get_fpl_players(season)

      matched_fpl_ids = set()

      # FBref is the source of truth
      for _, player_data in fbref_data.iterrows():
        player_id = player_data['id']
        player_name = player_data['name']

        # Ensure player entry is initialized
        self.player_dict.setdefault(player_id, {
          "FBref": {},
          "FPL": {}
        })

        # By default add FBref 
        self.player_dict[player_id]["FBref"] = {
          "id": player_id,
          "name": player_name
        }

        fpl_closest_match = self.find_fpl_closest_match(player_data, fpl_data, season)
        
        if fpl_closest_match is None or fpl_closest_match.empty:
          continue

        fpl_id = fpl_closest_match["id"]
        matched_fpl_ids.add(fpl_id)

        self.player_dict[player_id]['FPL'][season] = {
          "id": fpl_id,
          "name": fpl_closest_match['name']
        }

      # Add remaining FPL players that have no FBref match
      for _, fpl_player in fpl_data.iterrows():
        if fpl_player["id"] in matched_fpl_ids:
          continue

        print(f'No match for {fpl_player['name']} in {season}')

        fpl_name = fpl_player["name"]
        existing_name = self.find_existing_player(fpl_name, season)

        if existing_name:
          player_name = existing_name
        else:
          continue

        self.player_dict.setdefault(fpl_name, {
          "FPL": {}
        })
        self.player_dict[fpl_name]["FPL"][season] = {
          "id": fpl_player["id"],
          "name": player_name
        }

  def find_existing_player(self, name, season):
    """Finds the closest existing player name in player_dict by checking stored names and past FPL season names."""
    existing_names = list(self.player_dict.keys())

    # Also check all FPL names across seasons
    for player_key, player_data in self.player_dict.items():
      if season in player_data.get("FPL", '').keys():
        continue

      for season_data in player_data.get("FPL", {}).values():
        existing_names.append(season_data["name"]) 

    # Find the closest match among existing player names
    match_results = process.extractOne(name, existing_names, scorer=fuzz.token_sort_ratio)

    if match_results and match_results[1] >= 85:  # Stricter threshold to prevent mismatches
      matched_name = match_results[0]

      # Find which key in player_dict this name belongs to
      for player_key, player_data in self.player_dict.items():
        if player_key == matched_name:
          if season in player_data.get("FPL", '').keys():
            continue
          return player_key
    
    # No good match found
    return None

  def find_fpl_closest_match(self, fbref_player_data, fpl_data, season):
    # Find the closest match for a team name in a list of team names
    fb_name = fbref_player_data["name"]
    fb_goals = fbref_player_data["goals"]
    fb_assists = fbref_player_data["assists"]
    fb_teams = fbref_player_data["team"].split(";")

    team_mapping = TeamMatcher()
    fpl_team_ids = [
      team_data["FPL"][season]["id"] 
      for team_data in team_mapping.load_dict().values() 
      if team_data['FBref']['name'] in (fb_teams) and season in team_data["FPL"]
    ]
  
    fpl_data = fpl_data[fpl_data['team'].isin(fpl_team_ids)]

    # Find the best matching name in fpl_data
    match_results = process.extract(fb_name, fpl_data["name"], scorer=fuzz.token_set_ratio, limit=10)
    match_results = [(name, score) for name, score, _ in match_results if score >= 50]

    # If match goog enough
    if match_results and match_results[0][1] >= 70:
      best_match = fpl_data[fpl_data["name"] == match_results[0][0]].iloc[0]
      return best_match

    # Try if web name matches
    web_name_match_results = process.extract(fb_name, fpl_data["web_name"], scorer=fuzz.token_set_ratio, limit=10)
    web_name_match_results = [(name, score) for name, score, _ in web_name_match_results if score >= 70]
    
    if web_name_match_results and web_name_match_results[0][1] >= 70:
      best_match = fpl_data[fpl_data["web_name"] == web_name_match_results[0][0]].iloc[0]
      return best_match

    # No good match found
    if not match_results:
      return None 

    # Retrieve all candidate rows from FPL
    matched_candidates = fpl_data[fpl_data["name"].isin([match[0] for match in match_results])].copy()

    # Compute differences for goals and assists
    matched_candidates["goal_diff"] = abs(matched_candidates["goals"] - fb_goals)
    matched_candidates["assist_diff"] = abs(matched_candidates["assists"] - fb_assists)
    
    # Compute fuzzy name similarity score for sorting
    matched_candidates["name_similarity"] = matched_candidates["name"].apply(lambda x: fuzz.token_sort_ratio(fb_name, x))

    # Sort: Prioritize higher name similarity, then closest goal/assist match
    closest_match = matched_candidates.sort_values(by=["name_similarity", "goal_diff", "assist_diff"], ascending=[False, True, True]).iloc[0]
    return closest_match

  def get_fpl_player(self, key, season, key_type='id'):
    for _, player_data in self.player_dict.items():
      fpl_data = player_data.get("FPL", {})

      # Check if the season exists in FPL data
      if season in fpl_data:
        player_entry = fpl_data[season]

        # Check based on key_type
        if key_type in player_entry and player_entry[key_type] == key:
          return player_data

    raise Exception(f"Player could not be found for key {key} with type {key_type} in season {season}")

  def save_dict_to_json(self):
    # Save the dictionary to a JSON file
    with open(self.FILENAME, 'w') as file:
      json.dump(self.player_dict, 
        file, 
        indent=4, 
        default=lambda o: int(o) if isinstance(o, np.integer) else str(o))

  def load_dict_from_json(self):
    script_dir = os.path.dirname(os.path.abspath(__file__))    
    file_path = os.path.join(script_dir, self.FILENAME)
    
    with open(file_path, 'r', encoding='utf-8') as file:
      self.player_dict = json.load(file)
    
    return self.player_dict

if __name__ == "__main__":
  player_matcher = PlayerMatcher()
  player_matcher.create_player_dict()
  player_matcher.save_dict_to_json()