from data.fbref.data_loader import DataLoader as FBref

if __name__=='__main__':
  fbref = FBref('2019-20')
  fbref.get_player_match_logs(debug=True)
  # fbref.get_league_team_stats()
