import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import poisson
import json

# Charger les données préparées
def load_data():
    goal_model_data = pd.read_csv('matches1.csv')
    return goal_model_data

# Entraîner le modèle de Poisson
def train_poisson_model(goal_model_data):
    poisson_model = smf.glm(formula="goals ~ venue_int + team + opponent", data=goal_model_data, family=sm.families.Poisson()).fit()
    return poisson_model

# Fonction pour simuler un match et prédire le score
def simulate_match(poisson_model, home_team, away_team, max_goals=10):
    home_goals_avg = poisson_model.predict(pd.DataFrame(data={'team': home_team, 'opponent': away_team, 'venue_int': 1}, index=[1])).values[0]
    away_goals_avg = poisson_model.predict(pd.DataFrame(data={'team': away_team, 'opponent': home_team, 'venue_int': 0}, index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals + 1)] for team_avg in [home_goals_avg, away_goals_avg]]
    return np.outer(np.array(team_pred[0]), np.array(team_pred[1]))

# Fonction pour prédire le score d'un match
def predict(home_team, away_team, date):
    goal_model_data = load_data()
    poisson_model = train_poisson_model(goal_model_data)
    score_matrix = simulate_match(poisson_model, home_team, away_team, max_goals=5)
    
    home_score_pred = score_matrix.sum(axis=1).argmax()
    away_score_pred = score_matrix.sum(axis=0).argmax()

    result = {
        'date': date,
        'home_team': home_team,
        'away_team': away_team,
        'score_home_predicted': home_score_pred,
        'score_away_predicted': away_score_pred
    }

    results_list = [result]

    with open('matches.json', 'w') as f:
        json.dump(results_list, f, indent=4)

    print(results_list)

# Exemple d'utilisation
home_team = 'Manchester City'
away_team = 'Norwich City'
date = '2024-05-21'  # Exemple de date
predict(home_team, away_team, date)
home_team = 'Arsenal'
away_team = 'Norwich City'
date = '2024-05-28'  # Exemple de dates
predict(home_team, away_team, date)