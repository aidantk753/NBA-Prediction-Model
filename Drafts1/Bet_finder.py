import os
import requests
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo

load_dotenv()

api_key = os.getenv("ODDS_API_KEY")

url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"

# Defines a mapping of market keys to their corresponding abbreviations from Stat_finder.py
market_map={
    "player_points": "PTS",
    "player_rebounds": "REB",
    "player_assists": "AST",
    "player_steals": "STL",
    "player_blocks": "BLK"
}

def get_games(Date):
    response = requests.get(url, params={"apiKey": api_key})
    response.raise_for_status()
    games=response.json()           
    matching_games = []

    for game in games:
        game_date = datetime.fromisoformat(game["commence_time"].replace("Z", "+00:00")).astimezone(ZoneInfo("America/New_York")).date()
        if str(game_date) == Date:
            matching_games.append(game)

    return matching_games

def get_player_props(game_id):
    odds_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{game_id}/odds"

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": ",".join(market_map.keys()),
        "oddsFormat": "american"
    }

    odds_response = requests.get(odds_url, params=params)
    odds_response.raise_for_status()

    return odds_response.json() 


if __name__ == "__main__":
    Date = input("What date (YYYY-MM-DD)? ")
    matching_games = get_games(Date)
    if not matching_games:
        print(f"No games found for {Date}.")
        exit()

    # Checks if there are games on the specified date and displays them
    for game in matching_games:
        home_team = game["home_team"]
        away_team = game["away_team"]
        print(f"{home_team} vs {away_team}")

    # Prompt user for the team
    team=input("Which team? ").lower()

    # Check if the specified team has a game on the specified date and fetch odds for that game
    team_matches = [
        game for game in matching_games
        if team in game["home_team"].lower()
        or team in game["away_team"].lower()
    ]

    if not team_matches:
        print(f"No game found for {team} on {Date}.")
        exit()

    if len(team_matches) > 1:
        print("Multiple games match that team. Please be more specific:")
        for game in team_matches:
            print(f"{game['home_team']} vs {game['away_team']}")
        exit()

    game_id = team_matches[0]["id"]
    odds = get_player_props(game_id)

    # Prompt user for the bookmaker
    bookmaker_name = input("Enter the bookmaker name (enter 'all' for all bookmakers or 'best' for best odds): ")

    # Process the odds based on the user's input
    if bookmaker_name == "all":
        # Display the odds for all bookmakers
        for bookmaker in odds["bookmakers"]:
            print(f"Bookmaker: {bookmaker['title']}")
            for market in bookmaker["markets"]:
                print(f"Market: {market['key']}")
                for outcome in market["outcomes"]:
                    print(f"{outcome['description']} {outcome['name']} {outcome['point']}: {outcome['price']}")

        analysis = input("Statistical Line Analysis (y/n)? ")
        if analysis.lower() == "y":
            for bookmaker in odds["bookmakers"]:
                for market in bookmaker["markets"]:
                    for outcome in market["outcomes"]:
                        player_name = outcome["description"]
                        betting_line = outcome["point"]
                        stat_category = market_map[market["key"]]
        else:
            print("Statistical Line Analysis skipped.")
    # This is where you would call the function from Stat_finder.py to get the statistical line analysis for the player and stat category






    #
    elif bookmaker_name == "best":
        # Find the best odds for each market across all bookmakers
        best_odds = {}
        for bookmaker in odds["bookmakers"]:
            for market in bookmaker["markets"]:
                for outcome in market["outcomes"]:
                    bet_key = (
                        outcome['description'],
                        outcome['name'],
                        outcome['point'],
                        market['key']
                    )
                    if bet_key not in best_odds or outcome['price'] > best_odds[bet_key]["price"]:
                        best_odds[bet_key] = {"price": outcome['price'], "bookmaker": bookmaker['title']}
        # Display the best odds for each market
        for (description, side, point, market_key), odds_info in best_odds.items():
            print(f"Market: {market_key}")
            print(f"{description} {side} {point}: {odds_info['price']} ({odds_info['bookmaker']})")
        analysis=input("Statistical Line Analysis (y/n)? ")
        if analysis.lower() == "y":
            for (description, side, point, market_key), odds_info in best_odds.items():
                player_name = description
                betting_line = point
                stat_category = market_map[market_key]
    # This is where you would call the function from Stat_finder.py to get the statistical line analysis for the player and stat category








    #
        else:
            print("Statistical Line Analysis skipped.")
    else:
        # Check if the specified bookmaker is available
        bookmaker_found = False

        # Display the odds for the specified bookmaker
        for bookmaker in odds["bookmakers"]:
            if bookmaker['title'].strip().casefold() == bookmaker_name.strip().casefold():
                bookmaker_found = True
                print(f"Bookmaker: {bookmaker['title']}")

                for market in bookmaker["markets"]:
                    print(f"Market: {market['key']}")

                    for outcome in market["outcomes"]:
                        print(f"{outcome['description']} {outcome['name']} {outcome['point']}: {outcome['price']}")

        if not bookmaker_found:
            print(f"No odds found for bookmaker {bookmaker_name}.")
            exit()

        analysis = input("Statistical Line Analysis (y/n)? ")

        if analysis.lower() == "y":
            for bookmaker in odds["bookmakers"]:
                if bookmaker["title"].strip().casefold() == bookmaker_name.strip().casefold():
                    for market in bookmaker["markets"]:
                        for outcome in market["outcomes"]:
                            player_name = outcome["description"]
                            betting_line = outcome["point"]
                            stat_category = market_map[market["key"]]

                            # This is where you would call the function
                            # from Stat_finder.py
        else:
            print("Statistical Line Analysis skipped.")
