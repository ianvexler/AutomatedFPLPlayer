import random
import numpy as np

BUDGET = 1000
PLAYERS = 11
POSITIONS = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
MAX_POSITIONS = {"FWD": 3, "MID": 4, "DEF": 3, "GK": 1}
MAX_PER_TEAM = 3

class TeamSelector:
  def __init__(self, data):
    self.data = self.set_data(data)
    self.best_result = 0
    self.concurrently_worse = 0

  # TODO: Necessary pre-processing
  def set_data(self, data):
    return data
  
  def get_best_team(self):
    teams = self.set_initial_population()
    self.evaluate_teams(teams)
    
    return self.best_result

  # -------- Initial Population -------- 
  # - Generates initial population
  #   * Encode individuals
  #   * If individual doesn't meet FPL criteria (max players per pos. & team, budget, etc.) re-generate
  def set_initial_population(self):
    population_size = self.data.shape[0]
    
    teams = []
    for i in range(population_size + 1):
        teams.append(self.set_initial_team())
        
    return teams

  def set_initial_team(self):
    selected_players = np.random.choice(len(self.data), PLAYERS, replace=False)
    team = np.zeros(len(self.data))
    team[selected_players] = 1
    
    if self.check_fpl_requirements(team):
      return team
    else:
      return self.set_initial_team()

  # -------- Team Evaluation --------
  def evaluate_teams(self, teams):
    team_values = [self.calc_team_value(team) for team in teams]

    # Sorts the team values from best to worst
    sorted_indices = np.argsort(team_values)[::-1]
    ranked_teams = [teams[i] for i in sorted_indices]
    
    if self.check_satisfiability(ranked_teams[0]):
      return
    else:
      new_teams = self.set_new_team_generation(np.array(teams))
      return self.evaluate_teams(new_teams)
  
  # -------- Check Satisfiability --------
  # - Does it satisfy condition?
  #   * TBD - Currently checking if improvements on last 10 iterations
  def check_satisfiability(self, team) -> bool:    
    team_value = self.calc_team_value(team)
    
    if self.best_result < team_value:
      self.concurrently_worse = 0
      self.best_result = team_value
      self.best_team = team

      print(self.best_result)
    else:
      self.concurrently_worse += 1

    return False if self.concurrently_worse < 100 else True
  
  # -------- Generates new generation --------

  # - Define the generation of the new population
  #   * NOTES: The selection of which individuals from the current generation to replace can be based on various strategies 
  #     such as generational replacement, steady-state replacement, or other techniques.
  #   * Selects & mates parents
  #   * Does spring crossover & mutation
  def set_new_team_generation(self, teams):
    # self.select_parent_teams()
    new_teams = np.empty((0, teams.shape[1])) 
    
    while teams.shape[0] - 1 > 0:
      # Randomly select two teams, remove them from the list and breed them
      values = [self.calc_team_value(team) for team in teams]
      weights = 1 / np.array(values)
      weights /= weights.sum()

      random_teams = np.random.choice(teams.shape[0], size=2, replace=False, p=weights)
      selected_parent_teams = [teams[i] for i in random_teams]
      teams = np.delete(teams, random_teams, axis=0)
      
      # Add the removed item to the new array
      child_teams = self.breed_teams(selected_parent_teams)
      new_teams = np.append(new_teams, child_teams, axis=0)

    return new_teams
  
  # - Mate parents
  #   * Apply crossover and mutation on parents
  def breed_teams(self, parent_teams):
    child_teams = []
    for i in range(2):
      child_team = self.team_crossover(parent_teams)
      # child_team = self.team_mutation(child_team)
      child_teams.append(child_team)

    return child_teams
    
  # - Define the crossover:
  #   * TBD
  #   * One points crossover: pick a random crossover point
  #   * Uniform crossover: selecting genes on equal probability
  #   * Heuristic approach?
  #   * If individual doesn't meet FPL criteria (max players per pos. & team, budget, etc.) re-generate
  def team_crossover(self, parent_teams):
    genes = self.randomize_genes(parent_teams)
    child_team = np.zeros(len(self.data))
    child_team[genes] = 1

    if self.check_fpl_requirements(child_team):
      return child_team
    else:
      return self.team_crossover(parent_teams)
  
  # Concats the players from both teams and selects 11 genes at random
  # Genes are selected by weighted probability
  def randomize_genes(self, parent_teams):
    players_1 = np.where(parent_teams[0] == 1)[0]
    players_2 = np.where(parent_teams[1] == 1)[0]
    players_indexes = np.concatenate((players_1, players_2), axis=0)
    players_indexes = np.unique(players_indexes)

    players = self.data[players_indexes]
    points = [player[3] for player in players]
    weights = 1 / np.array(points)
    weights /= weights.sum()

    indexes = np.random.choice(players.shape[0], size=11, replace=False, p=weights)
    genes = np.array([players_indexes[i] for i in indexes])

    return genes

  # - Define the mutation:
  #   * Pick a selected player at random (maybe weighted probablity) and replace it with another (at complete random?)
  #   * If individual doesn't meet FPL criteria (max players per pos. & team, budget, etc.) re-generate
  def team_mutation(self, child_team):
    mutations = random.randint(0, 4)
    child_team[child_team == 1][:mutations] = 0
    child_team[child_team == 0][:mutations] = 1

    if self.check_fpl_requirements(child_team):
      return child_team
    else: 
      return self.team_mutation(child_team)

  def testing_team(self):
    selected_players = np.random.choice(len(self.data), PLAYERS, replace=False)
    print(selected_players)
  
  # -------- Utils --------
  # Recursion error
  def check_fpl_requirements(self, team) -> bool:
    selected_teams = {key: 0 for key in range(1, 21)}
    selected_positions = {key: 0 for key in range(1, 5)}
    team_cost = 0
        
    selected_players = np.where(team == 1)[0]
    reduced_players_data = self.data[selected_players]
    
    for player_data in reduced_players_data:
      selected_teams[player_data[1]] += 1
      selected_positions[player_data[2]] += 1
      
      position = POSITIONS[player_data[2]]
      position_count = MAX_POSITIONS[position]
      team_cost += player_data[3]
      
      # Checks for the three fpl rules
      if selected_teams[player_data[1]] > MAX_PER_TEAM:
        # print("Condition 1: selected_teams[player_data[1]] > MAX_PER_TEAM")
        return False

      if selected_positions[player_data[2]] > position_count:
        # print("Condition 2: selected_positions[player_data[2]] > position_count")
        return False

      if team_cost > BUDGET:
        # print("Condition 3: team_cost > BUDGET")
        return False

      if len(team[team == 1]) != PLAYERS:
        # print("Condition 4: len(team[team == 1]) != PLAYERS")
        return False
    
    if not all(value >= 1 for value in selected_positions.values()):
        # print(selected_positions)
        # print("Condition 5: At least one of each")
        return False
    
    return True

  # -------- Team Value --------
  # - Defines the fitness function
  #   * Based on how much the team would score
  #   * Fitness value: total score
  def calc_team_value(self, team):
    selected_players = np.where(team == 1)[0]
    reduced_players_data = self.data[selected_players]

    fitness_value = 0
    for player in reduced_players_data:
      fitness_value += player[3]
  
    return fitness_value