import streamlit as st
import football_app_full as fa  # Importing your football analysis code
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit app configuration
st.set_page_config(page_title="Football Data Analysis Dashboard", layout="wide")

# Title and description
st.title("Football Data Analysis Dashboard")
st.markdown("""
    Welcome to the Football Data Analysis Dashboard! Use the sidebar to select an analysis option and explore football statistics with real-time data visualizations.
""")

# Sidebar for navigation
st.sidebar.header("Analysis Options")
option = st.sidebar.selectbox(
    "Choose an analysis to perform:",
    [
        "Visualize Shot Map",
        "Enhanced Win Probability Model",
        "Goal Prediction Model",
        "Form Index",
        "Head to Head",
        "Travel Fatigue Impact",
        "Pass Accuracy Analysis",
        "Shot Conversion Rate",
        "Defensive Performance",
        "Player Heatmap",
        "Goalkeeper Performance",
        "Opponent Weaknesses",
        "Substitution Impact",
        "Dynamic Formation Changes",
        "Average Speed",
        "Crossing Accuracy",
        "Shooting Accuracy"
    ]
)

# Load data from Excel files (assuming .xlsx files are in the same directory)
try:
    fa.df_matches = pd.read_excel("matches.xlsx")
    fa.df_players = pd.read_excel("players.xlsx")
    fa.df_teams = pd.read_excel("teams.xlsx")
except FileNotFoundError:
    st.error("One or more Excel files not found. Please ensure 'matches.xlsx', 'players.xlsx', and 'teams.xlsx' are in the directory.")
    st.stop()

# Clean data
fa.df_matches = fa.clean_data(fa.df_matches)
fa.df_players = fa.clean_data(fa.df_players)
fa.df_teams = fa.clean_data(fa.df_teams)

# Function to display matplotlib/seaborn plots in Streamlit
def display_plot(func, param=None):
    with st.spinner("Generating visualization..."):
        if param:
            func(param)
        else:
            func()
        st.pyplot(plt.gcf())
        plt.clf()  # Clear the current figure to avoid overlap

# Main content based on selected option
if option == "Visualize Shot Map":
    player = st.text_input("Enter player name:", "Lionel Messi")
    if st.button("Generate Shot Map"):
        display_plot(fa.visualize_shot_map, player)

elif option == "Enhanced Win Probability Model":
    team = st.text_input("Enter team name (leave blank for all teams):", "")
    if st.button("Analyze Win Probability"):
        if team.strip() == "":
            result = fa.enhanced_win_probability_model()
        else:
            result = fa.enhanced_win_probability_model(team)
        st.pyplot(plt)
        st.write(result)

elif option == "Goal Prediction Model":
    if st.button("Run Model"):
        display_plot(fa.goal_prediction_model)

elif option == "Form Index":
    team = st.text_input("Enter team name:", "Manchester City")
    if st.button("Calculate Form Index"):
        display_plot(fa.form_index, team)

elif option == "Head to Head":
    team_input = st.text_input("Enter two team names (e.g., Team1,Team2):", "Manchester City,Liverpool")
    if st.button("Compare Teams"):
        fa.head_to_head(team_input)
        st.write(fa.df_matches[['match_date', 'home_team', 'away_team', 'home_team_goal', 'away_team_goal']])


elif option == "Travel Fatigue Impact":
    team = st.text_input("Enter team name:", "Chelsea")
    if st.button("Analyze Travel Impact"):
        display_plot(fa.travel_fatigue_impact, team)

elif option == "Pass Accuracy Analysis":
    if st.button("Analyze Pass Accuracy"):
        display_plot(fa.pass_accuracy_analysis)

elif option == "Shot Conversion Rate":
    if st.button("Analyze Conversion Rate"):
        display_plot(fa.shot_conversion_rate)

elif option == "Defensive Performance":
    if st.button("Analyze Defensive Stats"):
        display_plot(fa.defensive_performance)

elif option == "Player Heatmap":
    player = st.text_input("Enter player name:", "Kylian Mbappe")
    if st.button("Generate Heatmap"):
        display_plot(fa.player_heatmap, player)

elif option == "Goalkeeper Performance":
    if st.button("Analyze Goalkeeper Stats"):
        display_plot(fa.goalkeeper_performance)

elif option == "Opponent Weaknesses":
    team = st.text_input("Enter team name:", "Arsenal")
    if st.button("Identify Weaknesses"):
        display_plot(fa.opponent_weaknesses, team)

elif option == "Substitution Impact":
    if st.button("Analyze Substitution Impact"):
        display_plot(fa.substitution_impact)

elif option == "Dynamic Formation Changes":
    if st.button("Analyze Formation Changes"):
        display_plot(fa.dynamic_formation_changes)

elif option == "Average Speed":
    player = st.text_input("Enter player name:", "Neymar")
    if st.button("Calculate Speed"):
        result = fa.average_speed(player)
        st.write(result)

elif option == "Crossing Accuracy":
    if st.button("Analyze Crossing Accuracy"):
        display_plot(fa.crossing_accuracy)

elif option == "Shooting Accuracy":
    if st.button("Analyze Shooting Accuracy"):
        display_plot(fa.shooting_accuracy)

# Footer
st.markdown("---")
