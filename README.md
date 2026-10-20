NBA Player Performance Prediction & Backtesting Model:

  A python-based statistical modeling and backtesting tool designed to provide predictions of a players performance on any given night.
  The project aims to output a better-than-average prediction of a players points, rebounds, assists, steals, and blocks using career data
  
  Each test is performed and compared to a baseline in order to implement systematic optemization
  Ultimately, this model will investigate and expose the factors most relevant to a players statistical output


Project Overview:

  Using the most recent three seasons for all players with relevant game-data available on the NBA_API the model seeks to beat a players career average baseline
  For every game the model will treat the player's career average as a predictor of their output and the model will "guess" over/under based on its parameters
  From there, the model stores whether the prediction was correct and if not, to what numerical degree?
  Then it calculates a guessing percentage and a Mean Absolute Error
  If a more accurate model arises, said model becomes the new baseline and further refinements are tested
  
    Core areas of development:
      NBA player-statistic forecasting
      Weighted statistical modeling
      Historical backtesting and validation
      Parameter and weight optimization
      Injury probability and playing-time adjustments
      Matchup and pace analysis
      Prediction error analysis
  
  For every parameter added the model tests various weights in combination with the current parameters for best accuracy
  Weight Testing Comprises of:
    Alternative weight distributions
    Different recent-game sample windows
    Performance across statistical categories
    Prediction accuracy across historical evaluation periods
  
  Backtesting establishes the models performance against past games
  For each historical game the model:
    Establishes a prediction cutoff date.
    Retrieves player statistics available before that date.
    Generates a prediction using the selected model.
    Compares the prediction against the actual game result.
    Records and analyzes prediction errors.
    
Current Parameters:

  Injury and Playing-time models
    Injuries and playing times are intrinsically tied to opportunity and opportunity is the greatest indicator of success
    therefore, predicting these accurately was the first hurdle

    The project includes experimental components for:
      Modeling player availability probabilities
      Analyzing injury status and historical participation
      Estimating injury-related minutes adjustments

  Scoring: 5. Matchup and Pace Analysis
    For each game, the teams average pace and the defenses average allowed pace play a large part in how many touches a player may have
    the scoring component is where incorporation and testing begins

    Metrics tested:
      Opponent and matchup characteristics
      Game pace
      Shot volume
      Two-point and three-point attempts
      Free-throw opportunities
      Scoring-component errors


Technologies:
  Python
  Pandas
  NumPy
  NBA_API
  Github
  requests API

Repository Structure:
  The primary development scripts are located in Drafts1/.
    Weighted_model.py
      Weighted player-performance prediction model
    Weighted_shooting_model.py
      Experimental shooting-based prediction model
    backtest.py
      Historical prediction evaluation
    league_backtest.py
      League-wide backtesting
    optimize_weights.py
      Weight optimization experiments
    optimize_all_stats.py
      Optimization across statistical categories
    rank_weights.py
      Analysis and ranking of tested weight configurations
    injury_probability.py
      Player availability modeling
    injury_minutes_adjustment.py
      Injury-related playing-time adjustments
    pace_adjustments.py
      Pace-related modeling experiments
    validate_all_stats.py
      Historical model validation
    nba_cache_utils.py
      Local NBA data caching utilities


Model Evaluation:
  Model development follows an iterative process:
    Baseline prediction → Historical validation → Error analysis → Parameter adjustment → Out-of-sample evaluation


Current Development:
  This is an ongoing research and development project.
  Current areas of investigation include:
    Improving predictive accuracy across player-statistic categories
    Refining recent, seasonal, and career weight selection
    Evaluating injury-related adjustments
    Incorporating additional matchup information
    Testing alternative scoring-based prediction methods
    Improving the reliability of historical validation

