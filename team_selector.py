import numpy as np

BUDGET = 1000
PLAYERS = 16
POSITIONS = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
MAX_POSITIONS = {"FWD": 4, "MID": 5, "DEF": 6, "GK": 2}
MAX_PER_TEAM = 3

class TeamSelector:
  def __init__(self, data):
    self.data = self.set_data(data)
    self.best_result = 0
    self.concurrently_worse = 0

  def set_data(self, data):
    columns_to_extract = ['id', 'team', 'element_type', 'total_points', 'now_cost']
    column_names = data[0, :]
    column_indices = [np.where(column_names == col)[0][0] for col in columns_to_extract]
    reduced_data = data[1:, column_indices].astype(int)
    return reduced_data
  
  def get_best_team(self):
    teams = self.set_initial_population()

    return self.evaluate_teams(teams)

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
      return ranked_teams[0]
    else:
      return self.set_new_team_generation(teams)
    
  # -------- Team Value --------
  # - Defines the fitness function
  #   * Based on how much the team would score
  #   * Fitness value: total score
  def calc_team_value(self, team):
    selected_players = np.where(team == 1)
    reduced_players_data = self.data[selected_players]

    fitness_value = 0
    for player in reduced_players_data:
      fitness_value += player[3]
  
    return fitness_value
  
  # -------- Check Satisfiability --------
  # - Does it satisfy condition?
  #   * TBD - Currently checking if improvements on last 10 iterations
  def check_satisfiability(self, team) -> bool:    
    team_value = self.calc_team_value(team)

    if self.best_result <= team_value:
      self.concurrently_worse = 0
      self.best_result = team_value
    else:
      self.concurrently_worse += 1


    return False if self.concurrently_worse < 10 else True
  
  # -------- Generates new generation --------
  # * Selects & mates parents
  # * Does spring crossover & mutation
  def set_new_team_generation(self, teams):

    self.select_parent_teams()
    new_teams = np.array([])

    while len(teams) > 0:
      # Randomly select two teams, remove them from the list and breed them
      random_teams = np.random.choice(len(teams), size=2, replace=False)
      parent_teams = teams[random_teams]
      teams = np.delete(teams, random_teams)
      
      # Add the removed item to the new array
      child_teams = self.breed_teams(parent_teams)
      new_teams = np.concatenate((new_teams, child_teams))

    return new_teams
  
  # - Defines selection of individuals
  #   * Two options
  #   * Elitist: pick the parents with highest fitness value
  #   * Fitness-Proportionate Selection: selects parents with a probability proportional to their fitness (probably this one)
  #   * NOTES: Multiple parents are selected, generating two child each
  def select_parent_teams(self):
    print('here')

  # - Mate parents
  #   * Apply crossover and mutation on parents
  def breed_teams(self, parent_teams):
    self.team_crossover
    self.team_mutation
    print('here')

  # - Define the crossover:
  #   * TBD
  #   * One points crossover: pick a random crossover point
  #   * Uniform crossover: selecting genes on equal probability
  #   * Heuristic approach?
  #   * If individual doesn't meet FPL criteria (max players per pos. & team, budget, etc.) re-generate
  def team_crossover(self):
    print('here')

  # - Define the mutation:
  #   * Pick a selected player at random (maybe weighted probablity) and replace it with another (at complete random?)
  #   * If individual doesn't meet FPL criteria (max players per pos. & team, budget, etc.) re-generate
  def team_mutation(self):
    print('here')

  # - Define the generation of the new population
  #   * NOTES: The selection of which individuals from the current generation to replace can be based on various strategies 
  #     such as generational replacement, steady-state replacement, or other techniques.
  def set_new_population(self):
    print('here')
  
  # -------- Utils --------
  def check_fpl_requirements(self, team) -> bool:
    selected_teams = {key: 0 for key in range(1, 21)}
    selected_positions = {key: 0 for key in range(1, 5)}
    team_cost = 0
        
    selected_players = np.where(team == 1)
    reduced_players_data = self.data[selected_players]
    
    for player_data in reduced_players_data:
      selected_teams[player_data[1]] += 1
      selected_positions[player_data[2]] += 1
      
      position = POSITIONS[player_data[2]]
      position_count = MAX_POSITIONS[position]
      team_cost += player_data[3]
      
      # Checks for the three fpl rules
      if (selected_teams[player_data[1]] > MAX_PER_TEAM) or (selected_positions[player_data[2]] > position_count) or (team_cost > 1000):
        return False
    
    return True
