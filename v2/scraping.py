import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

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

    goal_model_data.to_csv('goal_model_data.csv', index=False)
    print("Data scraped and saved to goal_model_data.csv")

def rolling_averages(group, cols, new_cols):
    group = group.sort_values('date')
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

