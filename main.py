from data.fbref.data_loader import DataLoader as FBref

if __name__=='__main__':
  fbref = FBref('2020-21')
  fbref.get_player_match_logs()
