import numpy as np

BUDGET = 1000
PLAYERS = 16
POSITIONS = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
MAX_POSITIONS = {"FWD": 4, "MID": 5, "DEF": 6, "GK": 2}
MAX_PER_TEAM = 3

class TeamSelector:
  def __init__(self, data):
    self.data = self.set_data(data)

  def set_data(self, data):
    columns_to_extract = ['id', 'team', 'element_type', 'total_points', 'now_cost']
    column_names = data[0, :]
    column_indices = [np.where(column_names == col)[0][0] for col in columns_to_extract]
    reduced_data = data[1:, column_indices].astype(int)
    return reduced_data

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
    
  # -------- Fitness Function --------

  # - Defines the fitness function
  #   * Based on how much the team would score
  #   * Fitness value: total score
  def fitness_function(self, team):
    selected_players = np.where(team == 1)
    reduced_players_data = self.data[selected_players]

    fitness_value = 0
    for player in reduced_players_data:
      fitness_value += player[3]

    return fitness_value
    

  # - Does it satisfy condition?
  #   * TBD
  def check_satisfiability(self) -> bool:
    return True
  
  # - Defines selection of individuals
  #   * Two options
  #   * Elitist: pick the parents with highest fitness value
  #   * Fitness-Proportionate Selection: selects parents with a probability proportional to their fitness (probably this one)
  #   * NOTES: Multiple parents are selected, generating two child each
  def select_parents(self):
    print('here')

  # - Mate parents
  #   * Apply crossover and mutation on parents
  def mate_parents(self, parent_1, parent_2):
    print('here')

  # - Define the crossover:
  #   * TBD
  #   * One points crossover: pick a random crossover point
  #   * Uniform crossover: selecting genes on equal probability
  #   * Heuristic approach?
  #   * If individual doesn't meet FPL criteria (max players per pos. & team, budget, etc.) re-generate
  def offspring_crossover(self):
    print('here')

  # - Define the mutation:
  #   * Pick a selected player at random (maybe weighted probablity) and replace it with another (at complete random?)
  #   * If individual doesn't meet FPL criteria (max players per pos. & team, budget, etc.) re-generate
  def offspring_mutation(self):
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
      
      if (selected_teams[player_data[1]] > MAX_PER_TEAM) or (selected_positions[player_data[2]] > position_count) or (team_cost > 100):
        return False
    
    return True
