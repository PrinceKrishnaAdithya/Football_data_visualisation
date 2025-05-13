import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from geopy.distance import great_circle
from scipy.spatial.distance import cosine

# Load dataset from Excel files (assuming .xlsx due to WPS Office)
try:
    df_matches = pd.read_excel("matches.xlsx")  # Match results
    df_players = pd.read_excel("players.xlsx")  # Player stats
    df_teams = pd.read_excel("teams.xlsx")      # Team details
except FileNotFoundError:
    print("Error: One or more Excel files not found. Please check file paths.")
    exit()

def clean_data(df):
    """Clean dataframe by removing missing values."""
    initial_rows = len(df)
    df.dropna(inplace=True)
    print(f"Cleaned data: Removed {initial_rows - len(df)} rows with missing values.")
    return df

# Apply cleaning to all dataframes
df_matches = clean_data(df_matches)
df_players = clean_data(df_players)
df_teams = clean_data(df_teams)

def visualize_shot_map(player):
    """Scatter plot of shot locations with field context and detailed stats."""
    player_shots = df_players[df_players['player_name'] == player]
    if player_shots.empty:
        return f"No data found for player: {player}. Check player name spelling."
    
    plt.figure(figsize=(12, 8))
    
    if 'shot_outcome' in player_shots.columns:
        goals = player_shots[player_shots['shot_outcome'] == 1]
        misses = player_shots[player_shots['shot_outcome'] == 0]
        plt.scatter(misses['shot_x'], misses['shot_y'], c='red', alpha=0.5, label='Missed Shots')
        plt.scatter(goals['shot_x'], goals['shot_y'], c='green', alpha=0.7, label='Goals')
    else:
        plt.scatter(player_shots['shot_x'], player_shots['shot_y'], c='red', alpha=0.5, label='Shots')
    
    plt.axvline(x=0, color='green', linestyle='--', label='Left Goal Line')
    plt.axvline(x=100, color='green', linestyle='--', label='Right Goal Line')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axhline(y=100, color='gray', linestyle='-', alpha=0.3)
    plt.plot([0, 16.5], [20, 20], 'b-', alpha=0.5)
    plt.plot([0, 16.5], [80, 80], 'b-', alpha=0.5)
    plt.plot([16.5, 16.5], [20, 80], 'b-', alpha=0.5)
    plt.plot([83.5, 100], [20, 20], 'b-', alpha=0.5)
    plt.plot([83.5, 100], [80, 80], 'b-', alpha=0.5)
    plt.plot([83.5, 83.5], [20, 80], 'b-', alpha=0.5)
    
    plt.title(f"Shot Map for {player}", fontsize=16, pad=15)
    plt.xlabel("Field Length (0 = Left Goal, 100 = Right Goal)", fontsize=12)
    plt.ylabel("Field Width (0 = Bottom, 100 = Top)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.tight_layout()
    
    total_shots = len(player_shots)
    stats = f"Total shots plotted for {player}: {total_shots}"
    if 'shot_outcome' in player_shots.columns:
        goals_scored = len(goals)
        success_rate = (goals_scored / total_shots) * 100 if total_shots > 0 else 0
        avg_distance = player_shots['shot_x'].apply(lambda x: min(x, 100 - x)).mean()
        stats = (f"Shot Statistics for {player}:\n"
                 f"  Total Shots: {total_shots}\n"
                 f"  Goals Scored: {goals_scored}\n"
                 f"  Shot Success Rate: {success_rate:.2f}%\n"
                 f"  Average Distance from Goal: {avg_distance:.2f} units")
    return stats

def enhanced_win_probability_model(team=None):
    df_matches['travel_distance'] = df_matches.apply(
        lambda row: great_circle(
            (df_teams[df_teams['team_name'] == row['home_team']]['latitude'].iloc[0],
             df_teams[df_teams['team_name'] == row['home_team']]['longitude'].iloc[0]),
            (df_teams[df_teams['team_name'] == row['away_team']]['latitude'].iloc[0],
             df_teams[df_teams['team_name'] == row['away_team']]['longitude'].iloc[0])
        ).km, axis=1
    )

    # Filter matches involving the specified team (as home or away)
    if team:
        df_filtered = df_matches[
            (df_matches['home_team'] == team) | (df_matches['away_team'] == team)
        ]
    else:
        df_filtered = df_matches

    if df_filtered.empty:
        return f"No matches found for team: {team}"

    features = ['possession', 'shots_on_target', 'pass_accuracy', 'home_team_goal', 'travel_distance']
    X = df_filtered[features]
    y = df_filtered['result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    coefficients = model.coef_[0]
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Coefficient': coefficients
    })
    feature_importance = feature_importance.sort_values(by='Coefficient', ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], color='teal')
    plt.xlabel('Coefficient Magnitude (Impact on Outcome)')
    plt.title(f'Feature Importance in Enhanced Win Probability Model for {team}' if team else 'Feature Importance in Enhanced Win Probability Model')
    for i, v in enumerate(feature_importance['Coefficient']):
        plt.text(v, i, f'{v:.3f}', va='center', ha='left' if v > 0 else 'right')

    return f"Model Accuracy: {accuracy:.2f}"

def goal_prediction_model():
    """Linear Regression to predict home team goals."""
    X = df_matches[['possession', 'shots_on_target', 'pass_accuracy']]
    y = df_matches['home_team_goal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = model.score(X_test, y_test)
    
    sample = [[50, 5, 85]]
    pred_goals = model.predict(sample)[0]
    
    output = (f"Goal Prediction Model - Mean Squared Error: {mse:.2f} (lower is better)\n"
              f"Goal Prediction Model - R² Score: {r2:.2f} (closer to 1 is better)\n"
              f"Model Coefficients (Impact on Goals):\n")
    for feature, coef in zip(X.columns, model.coef_):
        output += f"  {feature}: {coef:.3f}\n"
    output += f"Intercept: {model.intercept_:.3f}\n"
    output += f"\nSample Prediction (Possession=50%, Shots=5, Pass=85%): {pred_goals:.2f} goals"
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, c=np.where(predictions > y_test, 'orange', 'blue'), 
                alpha=0.5, label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.title("Goal Prediction: Actual vs Predicted\n(Blue: Underpredicted, Orange: Overpredicted)", fontsize=16)
    plt.xlabel("Actual Goals Scored", fontsize=12)
    plt.ylabel("Predicted Goals", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    return output

def form_index(team):
    """Calculate weighted moving average for last 5 matches."""
    team_matches = df_matches[df_matches['home_team'] == team]
    if team_matches.empty:
        return f"No data found for team: {team}"
    team_matches['form_index'] = team_matches['home_team_goal'].rolling(window=5, min_periods=1).mean()
    plt.figure(figsize=(10, 6))
    team_matches[['match_date', 'form_index']].set_index('match_date').plot(title=f"Form Index for {team}", fontsize=14)
    plt.xlabel("Match Date", fontsize=12)
    plt.ylabel("Form Index (Goals)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    return ""

def head_to_head(team_input=None):
    """Compare past matches between two teams."""
    if team_input is None:
        return "Please provide two team names separated by a comma (e.g., Team1,Team2)"
    try:
        team1, team2 = team_input.split(",")
        h2h_matches = df_matches[((df_matches['home_team'] == team1.strip()) & (df_matches['away_team'] == team2.strip())) |
                                 ((df_matches['home_team'] == team2.strip()) & (df_matches['away_team'] == team1.strip()))]
        if h2h_matches.empty:
            return f"No head-to-head data found for {team1} vs {team2}"
        return h2h_matches[['match_date', 'home_team', 'away_team', 'home_team_goal', 'away_team_goal']].to_string()
    except ValueError:
        return "Invalid input. Please provide two team names separated by a comma."


def travel_fatigue_impact(team):
    """Analyze travel impact on performance."""
    global df_matches
    team_matches = df_matches[df_matches['home_team'] == team]
    if team_matches.empty:
        return f"No data found for team: {team}"
    team_matches['travel_distance'] = team_matches.apply(lambda x: great_circle(
        (df_teams.loc[df_teams['team_name'] == x['home_team'], ['latitude', 'longitude']].values[0]),
        (df_teams.loc[df_teams['team_name'] == x['away_team'], ['latitude', 'longitude']].values[0])
    ).km, axis=1)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=team_matches['travel_distance'], y=team_matches['home_team_goal'])
    plt.title(f"Impact of Travel Distance on Goals for {team}", fontsize=14)
    plt.xlabel("Travel Distance (km)", fontsize=12)
    plt.ylabel("Home Team Goals", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    return ""

def pass_accuracy_analysis():
    """Analyze pass accuracy across teams."""
    pass_accuracy = df_players.groupby('team_name')['pass_accuracy'].mean()
    plt.figure(figsize=(10, 6))
    pass_accuracy.plot(kind='bar', title='Team Pass Accuracy', fontsize=14)
    plt.xlabel("Team Name", fontsize=12)
    plt.ylabel("Average Pass Accuracy", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    return ""

def shot_conversion_rate():
    """Analyze the shot conversion rate for teams, considering only players with assists."""
    global df_players
    
    # Filter players with at least one assist and create a copy to avoid modifying the original dataframe
    df_assist_players = df_players[df_players['shots'] > 0].copy()
    
    # Check if there are any players with assists; if not, return a message
    if df_assist_players.empty:
        return "No players with assists found in the data."
    
    # Calculate conversion rate, handling division by zero (shots = 0) by setting rate to 0
    df_assist_players['conversion_rate'] = df_assist_players.apply(
        lambda row: row['goals'] / row['shots'] if row['shots'] > 0 else 0, axis=1
    )
    
    # Create the boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df_assist_players['team_name'], y=df_assist_players['conversion_rate'])
    plt.title("Shot Conversion Rate for Players with Assists", fontsize=14)
    plt.xlabel("Team Name", fontsize=12)
    plt.ylabel("Conversion Rate", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    return ""
def defensive_performance():
    """Evaluate defensive performance by analyzing tackles and interceptions."""
    defensive_stats = df_players.groupby('team_name')[['tackles', 'interceptions']].mean()
    plt.figure(figsize=(10, 6))
    defensive_stats.plot(kind='bar', stacked=True, title='Defensive Performance', fontsize=14)
    plt.xlabel("Team Name", fontsize=12)
    plt.ylabel("Average Stats", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    return ""

def player_heatmap(player):
    player_moves = df_players[df_players['player_name'] == player]
    if player_moves.empty:
        return f"No data found for player: {player}. Check player name spelling."
    
    plt.figure(figsize=(12, 8))
    # Scatter plot for individual positions
    plt.scatter(player_moves['position_x'], player_moves['position_y'], c='blue', alpha=0.5, s=100, label='Positions')
    # KDE plot with adjusted bandwidth for better density
    sns.kdeplot(x=player_moves['position_x'], y=player_moves['position_y'], cmap='Reds', fill=True, alpha=0.6, bw_adjust=0.5)
    
    plt.axvline(x=0, color='green', linestyle='--', label='Left Goal Line')
    plt.axvline(x=100, color='green', linestyle='--', label='Right Goal Line')
    plt.plot([0, 16.5], [20, 20], 'b-', alpha=0.5)
    plt.plot([0, 16.5], [80, 80], 'b-', alpha=0.5)
    plt.plot([16.5, 16.5], [20, 80], 'b-', alpha=0.5)
    plt.plot([83.5, 100], [20, 20], 'b-', alpha=0.5)
    plt.plot([83.5, 100], [80, 80], 'b-', alpha=0.5)
    plt.plot([83.5, 83.5], [20, 80], 'b-', alpha=0.5)
    
    plt.title(f"Heatmap of {player}'s Movements", fontsize=16, pad=15)
    plt.xlabel("Field Length (0 = Left Goal, 100 = Right Goal)", fontsize=12)
    plt.ylabel("Field Width (0 = Bottom, 100 = Top)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.tight_layout()
    
    avg_x = player_moves['position_x'].mean()
    avg_y = player_moves['position_y'].mean()
    return (f"Movement Statistics for {player}:\n"
            f"  Average Position: ({avg_x:.2f}, {avg_y:.2f})\n"
            f"  - {avg_x:.2f} units from Left Goal, {100 - avg_x:.2f} units from Right Goal\n"
            f"  - {avg_y:.2f} units from Bottom, {100 - avg_y:.2f} units from Top") 

def goalkeeper_performance():
    """Analyze goalkeeper saves and clean sheets."""
    keeper_stats = df_players[df_players['position'] == 'Goalkeeper']
    if keeper_stats.empty:
        return "No goalkeeper data found."
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=keeper_stats['saves'], y=keeper_stats['clean_sheets'])
    plt.title("Goalkeeper Performance", fontsize=14)
    plt.xlabel("Saves", fontsize=12)
    plt.ylabel("Clean Sheets", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    return ""

def opponent_weaknesses(team):
    """Identify weaknesses in opponent teams."""
    weaknesses = df_matches[df_matches['away_team'] == team].groupby('home_team')['goals_conceded'].mean()
    if weaknesses.empty:
        return f"No data found for team: {team}"
    plt.figure(figsize=(10, 6))
    weaknesses.plot(kind='bar', title=f'Opponent Weaknesses Against {team}', fontsize=14)
    plt.xlabel("Opponent Team", fontsize=12)
    plt.ylabel("Average Goals Conceded", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    return ""

def substitution_impact():
    """Evaluate the impact of substitutions on match results."""
    impact = df_players.groupby('substitution_time')['impact_rating'].mean()
    plt.figure(figsize=(10, 6))
    impact.plot(title='Substitution Impact', fontsize=14)
    plt.xlabel("Substitution Time", fontsize=12)
    plt.ylabel("Average Impact Rating", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    return ""

def dynamic_formation_changes(team_pair=None):
    # If team_pair is provided, filter matches between the two teams
    if team_pair:
        try:
            team1, team2 = [t.strip() for t in team_pair.split(",")]
        except ValueError:
            return "Please enter two team names separated by a comma (e.g., 'Chelsea,Arsenal')."

        df_filtered = df_matches[
            ((df_matches['home_team'] == team1) & (df_matches['away_team'] == team2)) |
            ((df_matches['home_team'] == team2) & (df_matches['away_team'] == team1))
        ]
        title = f"Impact of Formation Changes for {team1} vs {team2}"
    else:
        df_filtered = df_matches
        title = "Impact of Formation Changes (All Teams)"

    if df_filtered.empty:
        return f"No matches found between teams: {team1} and {team2}"

    # Filter to include only rows with formation changes (containing "to")
    df_filtered = df_filtered[df_filtered['formation_change'].str.contains("to", case=False, na=False)]

    # Check if there are any matches with formation changes
    if df_filtered.empty:
        return "No formation changes found for the selected teams."

    # Group by formation_change and calculate average result
    formation_impact = df_filtered.groupby('formation_change')['result'].mean().sort_values()

    # Plot the results
    plt.figure(figsize=(10, 6))
    formation_impact.plot(kind='bar', color='teal')
    plt.title(title)
    plt.xlabel("Formation Change")
    plt.ylabel("Average Result (0=Draw, 1=Home Win, 2=Away Win)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    return ""

def average_speed(player):
    player_data = df_players[df_players['player_name'] == player]
    if player_data.empty:
        return f"No data found for player: {player}"
    avg_speed = player_data['speed'].mean()
    return f"Average Speed of {player}: {avg_speed:.2f} km/h"

def crossing_accuracy():
    """Analyze crossing accuracy of players."""
    crossing_stats = df_players.groupby('team_name')['crossing_accuracy'].mean()
    plt.figure(figsize=(10, 6))
    crossing_stats.plot(kind='bar', title='Crossing Accuracy by Team', fontsize=14)
    plt.xlabel("Team Name", fontsize=12)
    plt.ylabel("Average Crossing Accuracy", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    return ""

def shooting_accuracy():
    """Analyze shooting accuracy of players."""
    shooting_stats = df_players.groupby('team_name')['shooting_accuracy'].mean()
    plt.figure(figsize=(10, 6))
    shooting_stats.plot(kind='bar', title='Shooting Accuracy by Team', fontsize=14)
    plt.xlabel("Team Name", fontsize=12)
    plt.ylabel("Average Shooting Accuracy", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    return ""

def menu():
    """Interactive menu with consistent structure."""
    functions = {
        '1': ('Visualize Shot Map', visualize_shot_map, 'Enter player name: '),
        '2': ('Enhanced Win Probability Model (Logistic)', enhanced_win_probability_model, None),
        '3': ('Form Index', form_index, 'Enter team name: '),
        '4': ('Head to Head', head_to_head, 'Enter two team names separated by a comma (e.g., Team1,Team2): '),
        '5': ('Player Consistency', player_consistency, 'Enter player name: '),
        '6': ('Travel Fatigue Impact', travel_fatigue_impact, 'Enter team name: '),
        '7': ('Pass Accuracy Analysis', pass_accuracy_analysis, None),
        '8': ('Shot Conversion Rate', shot_conversion_rate, None),
        '9': ('Defensive Performance', defensive_performance, None),
        '10': ('Player Heatmap', player_heatmap, 'Enter player name: '),
        '11': ('Goalkeeper Performance', goalkeeper_performance, None),
        '12': ('Opponent Weaknesses', opponent_weaknesses, 'Enter team name: '),
        '13': ('Substitution Impact', substitution_impact, None),
        '14': ('Dynamic Formation Changes', dynamic_formation_changes, None),
        '15': ('Average Speed', average_speed, 'Enter player name: '),
        '16': ('Crossing Accuracy', crossing_accuracy, None),
        '17': ('Shooting Accuracy', shooting_accuracy, None),
        '18': ('Goal Prediction Model (Linear)', goal_prediction_model, None),
    }
    
    while True:
        print("\n=== Football Data Analysis Menu ===")
        for key, (name, _, _) in functions.items():
            print(f"{key}. {name}")
        print("19. Exit")
        
        choice = input("Enter choice: ")
        if choice == '19':
            print("Exiting program. Goodbye!")
            break
        elif choice in functions:
            name, func, prompt = functions[choice]
            try:
                if prompt:
                    param = input(prompt)
                    result = func(param)
                    if result:
                        print(result)
                else:
                    result = func()
                    if result:
                        print(result)
            except Exception as e:
                print(f"Error occurred in {name}: {str(e)}")
        else:
            print("Invalid choice. Please select a number between 1 and 19.")

if __name__ == "__main__":
    menu()