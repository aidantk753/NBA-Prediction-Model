from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import statistics

#Prompt User for player name
player_name = input("Enter the player's name (First Last): ")

#prompt User for stat category
stat_category = input("Enter the stat category (e.g., PTS, AST, REB, STL, BLK): ")
stat_category = stat_category.strip().upper()

if stat_category not in ["PTS", "AST", "REB", "STL", "BLK"]:
    print("Invalid statistic.")
    exit()

# Function to get player ID based on name
matching_players = players.find_players_by_full_name(player_name)

if not matching_players:
    print(f"No player found matching '{player_name}'.")
    exit()

player_id = matching_players[0]['id']

# Fetch player career stats
career_stats = playercareerstats.PlayerCareerStats(player_id).get_data_frames()[1]

if career_stats.empty:
    print(f"No career statistics available for {player_name}.")
    exit()

stat = career_stats.loc[0, stat_category]
gp = career_stats.loc[0, 'GP']

if gp == 0:
    print(f"No career games available for {player_name}.")
    exit()

# Calculate career stat per game
stat_per_game = stat / gp

# Fetch player game logs
game_logs = playergamelog.PlayerGameLog(
    player_id=player_id,
    season="2026-27"
).get_data_frames()[0]
stats_list = game_logs[stat_category].tolist()

# Calculate the season average and standard deviation for the requested stat
if len(stats_list) == 0:
    print("No games available.")
    exit()

season_average = sum(stats_list) / len(stats_list)

if len(stats_list) > 1:
    std_dev = statistics.pstdev(stats_list)
else:
    std_dev = 0.0

# Display the result for mean and standard deviation
print(f"{player_name} has a career average of {stat_per_game:.2f} {stat_category} per game")
print(f"The seasonal mean and standard deviation for {stat_category} is {season_average:.2f} ± {std_dev:.2f}")

# Prompt user for the player's betting line for the requested stat
try:
    betting_line = float(input(f"Enter the betting line for {player_name}'s {stat_category}: "))
except ValueError:
    print("Invalid betting line. Please enter a number.")
    exit()
# Calculate z-score for the player's betting line
if std_dev != 0:
    z_score = (betting_line - season_average) / std_dev
    # Display the z-score for the betting line
    print(f"The z-score for the betting line is {z_score:.2f}")
else:
    print("Standard deviation is zero, cannot calculate z-score.")