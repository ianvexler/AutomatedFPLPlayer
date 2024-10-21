class Gameweek:
    def __init__(self, num, data):
        self.num = num
        self.data = data

    def get_player_data(self, player_name):
        return self.data[self.data[:, 0] == player_name]
