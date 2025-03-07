# AutomatedFPLPlayer

Project aimed at automating Fantasy Premier League (FPL) team selection using machine learning techniques. It focuses on optimizing player selection and transfers based on historical data, player performance metrics, and predictive models. The goal is to enhance decision-making and maximize points in FPL with minimal manual intervention.

</br>


# Key Classes

## Model Class - Description  

The `Model` class is the core of the **AutomatedFPLPlayer** project, responsible for training and predicting Fantasy Premier League (FPL) player performance using various machine learning models. It supports multiple regression models, including **Random Forest, AdaBoost, Gradient Boosting, XGBoost, and LSTM (single-layer & multi-layer)**.  

The `Model` class supports several command-line arguments to customize its behavior:  

| Argument       | Type   | Description  |
|---------------|--------|-------------|
| `--steps` | `int` | Number of time steps for the data window (default: `5`). Controls how many previous gameweeks are considered for predictions. |
| `--season` | `str` | The season to simulate, formatted as `20xx-yy` (default: `2024-25`). |
| `--prev_season` | `flag` | If enabled, includes data from the previous season for better long-term trend analysis. |
| `--model` | `str` | Specifies the model type to use. Options: `random_forest`, `adaboost`, `gradient_boost`, `xgboost`, `lstm`, `ml_lstm`. |
| `--no_train` | `flag` | If set, skips training and loads a pre-trained model instead. |

Predictions are automatically saved in the predictions folder

## Evaluate

The `Evaluate` class assesses the performance of trained models by comparing predicted player points with actual FPL data. It calculates various error metrics, groups evaluations based on player characteristics (e.g., points and cost), and exports results for further analysis.  

The `Evaluate` class supports several command-line arguments to customize its evaluation process:  

| Argument       | Type   | Description  |
|---------------|--------|-------------|
| `--season` | `str` | The season to evaluate, formatted as `20xx-yy` (default: `2024-25`). |
| `--steps` | `int` | Number of time steps used in training (default: `5`). Must match the model's configuration. |
| `--prev_season` | `flag` | If enabled, includes data from the previous season in evaluation. |
| `--model` | `str` | Specifies the model type to evaluate. Options: `random_forest`, `adaboost`, `gradient_boost`, `xg_boost`, `lstm`, `ml_lstm`. |

All evaluations must have an already existing prediction

## Simulation

## Simulation Class - Description  

The `Simulation` class models a full Fantasy Premier League (FPL) season, managing transfers, budget constraints, and chip strategies to optimize team performance. It simulates gameweeks (GWs) based on predicted player points, allowing for strategic decisions such as wildcard usage, free hit activation, and captain selection.  

The `Simulation` class supports several command-line arguments to customize the season simulation process:  

| Argument       | Type   | Description  |
|---------------|--------|-------------|
| `--season` | `str` | The season to simulate, formatted as `20xx-yy` (default: `2024-25`). |
| `--triple_captain` | `str` | Strategy for the Triple Captain chip. Options: `'double_gw'`, `'blank_gw'` (default: `'double_gw'`). |
| `--wildcard` | `str` | Strategy for the Wildcard chip. Options: `'double_gw'`, `'blank_gw'`, `'use_asap'`, `'wait'` (default: `'double_gw'`). |
| `--free_hit` | `str` | Strategy for the Free Hit chip. Options: `'double_gw'`, `'blank_gw'` (default: `'blank_gw'`). |
| `--bench_boost` | `str` | Strategy for the Bench Boost chip. Options: `'double_gw'`, `'blank_gw'` (default: `'double_gw'`). |
| `--max_gw` | `int` | The maximum gameweek to simulate (default: `38`). |

</br>

# Utils

## Team

The `Team` class is responsible for managing FPL squad selection, transfers, captaincy choices, and in-game substitutions. It optimizes player selection based on expected points (`xP`), budget constraints, and FPL-specific rules such as formation restrictions and team limits. The class integrates with the `Simulation` module to dynamically adjust teams across the season.  

The `Team` class is primarily used within the `Simulation` module.

## Team Matcher

The `TeamMatcher` class is responsible for mapping Fantasy Premier League (FPL) team names to external datasets such as **Club Elo** and **FBref**. It uses matching techniques to align team names across different sources, ensuring consistency in data integration.

## Player Matcher

The `PlayerMatcher` class maps Fantasy Premier League (FPL) player data to external datasets such as **FBref**. It ensures consistency in player identification by using matching techniques, aligning player names across different sources while considering team affiliations and performance statistics. This class is essential for integrating player data into predictive models.  

## Feature Scaler

The `FeatureScaler` class is responsible for scaling numerical features in the dataset to improve model performance. It applies different scaling techniques to normalize and standardize data, ensuring consistency in input values for machine learning models.

## Feature Selector

The `FeatureSelector` class is responsible for defining and selecting relevant features for training machine learning models in Fantasy Premier League (FPL) player performance prediction. It organizes features by player positions and ensures that key attributes such as goals, assists, clean sheets, and expected statistics are included.