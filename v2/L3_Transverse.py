import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
from sklearn.metrics import precision_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson

# Fonction pour charger les données et entraîner le modèle de Poisson
def load_and_prepare_data():
    standings_url = 'https://fbref.com/en/comps/9/2022-2023/2022-2023-Premier-League-Stats'
    years = list(range(2023, 2019, -1))
    all_matches = []

    for year in years:
        data = requests.get(standings_url)
        soup = BeautifulSoup(data.text, "html.parser")
        standings_table = soup.select('table.stats_table')[0]

        links = [l.get("href") for l in standings_table.find_all('a')]
        links = [l for l in links if '/squads/' in l]
        team_urls = [f"https://fbref.com{l}" for l in links]

        previous_season = soup.select('a.prev')[0].get('href')
        standings_url = f'http://fbref.com{previous_season}'
        
        for team_url in team_urls:
            team_name = team_url.split('/')[-1].replace('-Stats', '').replace('-', ' ')
            data = requests.get(team_url)
            matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
            soup = BeautifulSoup(data.text, features="lxml")
            links = [l.get("href") for l in soup.find_all('a')]
            links = [l for l in links if l and 'all_comps/shooting/' in l]
            data = requests.get(f"https://fbref.com{links[0]}")
            shooting = pd.read_html(data.text, match="Shooting")[0]
            shooting.columns = shooting.columns.droplevel()

            try:
                team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
            except ValueError:
                continue

            team_data = team_data[team_data['Comp'] == 'Premier League']
            team_data['Season'] = year
            team_data['Team'] = team_name
            all_matches.append(team_data)
            time.sleep(4)

    match_df = pd.concat(all_matches)
    match_df.columns = [c.lower() for c in match_df.columns]
    match_df['date'] = pd.to_datetime(match_df['date'])
    match_df['venue_code'] = match_df['venue'].astype('category').cat.codes
    match_df['opp_code'] = match_df['opponent'].astype('category').cat.codes
    match_df['hour'] = match_df['time'].replace(":.+", "", regex=True).astype('int')
    match_df['day_code'] = match_df['date'].dt.dayofweek
    match_df['target'] = (match_df['result'] == 'W').astype('int')

    matches_rolling = match_df.groupby('team').apply(lambda x: rolling_averages(x, ['gf', 'ga', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt'], [f'{c}_rolling' for c in ['gf', 'ga', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt']]))
    matches_rolling.index = range(matches_rolling.shape[0])
    matches_rolling['venue_int'] = matches_rolling['venue'].map({'Home': 1, 'Away': 0})

    goal_model_data = matches_rolling[['team', 'opponent', 'gf', 'venue_int']]
    goal_model_data.rename(columns={'gf': 'goals'}, inplace=True)
    goal_model_data['team'] = goal_model_data['team'].astype('category')
    goal_model_data['opponent'] = goal_model_data['opponent'].astype('category')
    goal_model_data['venue_int'] = goal_model_data['venue_int'].astype(int)
    goal_model_data['goals'] = goal_model_data['goals'].astype(float)

    poisson_model = smf.glm(formula="goals ~ venue_int + team + opponent", data=goal_model_data, family=sm.families.Poisson()).fit()
    return poisson_model

# Fonction pour simuler un match et prédire le score
def simulate_match(poisson_model, home_team, away_team, max_goals=10):
    home_goals_avg = poisson_model.predict(pd.DataFrame(data={'team': home_team, 'opponent': away_team, 'venue_int': 1}, index=[1])).values[0]
    away_goals_avg = poisson_model.predict(pd.DataFrame(data={'team': away_team, 'opponent': home_team, 'venue_int': 0}, index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals + 1)] for team_avg in [home_goals_avg, away_goals_avg]]
    return np.outer(np.array(team_pred[0]), np.array(team_pred[1]))

# Fonction pour prédire le score d'un match
def predict(home_team, away_team):
    poisson_model = load_and_prepare_data()
    score_matrix = simulate_match(poisson_model, home_team, away_team, max_goals=5)

    # Créez une instance de votre modèle de prédiction
    model = RandomForestClassifier(
        n_estimators=100,    # Nombre d'arbres dans la forêt
        criterion='gini',    # Fonction pour mesurer la qualité d'une scission
        max_depth=None,      # Profondeur maximale de l'arbre, None signifie que les nœuds sont développés jusqu'à ce que toutes les feuilles soient pures
        min_samples_split=2, # Le nombre minimal d'échantillons requis pour scinder un nœud interne
        min_samples_leaf=1,  # Le nombre minimal d'échantillons requis pour être à un nœud feuille
        random_state=42      # Une graine pour la reproductibilité des résultats
    )


    return score_matrix

# Exemple d'utilisation
home_team = 'Manchester City'
away_team = 'Norwich City'
score_matrix = predict(home_team, away_team)
print(score_matrix)
