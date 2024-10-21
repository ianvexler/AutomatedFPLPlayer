import numpy as np

class Player:
  def __init__(self, name, data, raw):
    positions = {'1': 'GK', '2': 'DEF', '3': 'MID', '4': 'FWD'}

    self.name = name
    self.data = data
    self.raw = raw

    self.position = positions[raw[20]]
    self.cost = int(raw[50])

  def get_gw_stat(self, gw, metric):
    # Gets the index of the desired metric
    index = np.where(self.data[0] == metric)[0][0]
    return self.data[gw, index]
  
  def get_raw_stat(self, metric):
    index = np.where(self.raw[0] == metric)[0][0]
    return self.raw[index]
  