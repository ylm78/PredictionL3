#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests #On importe le module requests pour pouvoir faire des requêtes HTTP


# In[2]:


standings_url = 'https://fbref.com/en/comps/9/2022-2023/2022-2023-Premier-League-Stats' #L'url avec laquelle on va récupérer le classement de premire leag


# In[3]:


data = requests.get(standings_url, headers = {"Accept" : "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
"User-Agent" : "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0"
})


# In[4]:


data.text #On affiche la réponse en fesant .text sinon on obtient uniquement le status code


# In[5]:


from bs4 import BeautifulSoup #On importe BeautifulSoup


# In[6]:


soup = BeautifulSoup(data.text) #On transforme le contenu de notre page en objet BeautifulSoup


# In[7]:


standings_table = soup.select("table.stats_table")[0] #On selectionne que le premier tableau "Overall" parmi tout les tableaux de class = "stats_table"


# In[8]:


links = standings_table.find_all('a') #On stock dans une liste tout les éléments avec une balise a find_all RETOURNE UN TABLEAU


# In[9]:


links = [l.get ('href') for l in links] #On stock le href de chaque élément a en tant qu'élément unique dans la liste links


# In[10]:


links = [l for l in links if 'squads' in l] #On stock tout les liens (href) contenant '/squads/'


# In[11]:


links


# In[12]:


len(links) #On voit bien qu'on a 20 équipes


# In[13]:


team_urls = [f'https://fbref.com/{l}' for l in links] #On complete les liens de chaque équipe avec le préfixe identique


# In[14]:


team_urls


# In[15]:


team_url =team_urls[0] #On prend man city par exemple


# In[16]:


team_url


# In[17]:


data = requests.get(team_url) #On fait une requête get vers le lien ci-dessus


# In[18]:


data.text #.text sinon on aura le status code


# In[19]:


import pandas as pd #Import de la librairie pandas pour manipuler les données


# In[20]:


matches = pd.read_html(data.text, match="Scores & Fixtures") #read_html retourne une liste


# In[21]:


matches


# In[22]:


matches[0] #Notre liste contient UNE seule DF donc on l'extrait


# In[23]:


soup = BeautifulSoup(data.text) #soup devient un objet BeautifulSoup que l'on peut scrapper


# In[24]:


links = soup.find_all('a') #On récupére toutes les balises a dans une list qui s'appelle links


# In[25]:


links


# In[26]:


links = [l.get('href') for l in links] #On récupère le lien (href) de chaque élément de la liste links


# In[27]:


links


# In[28]:


links=[l for l in links if l and 'all_comps/shooting' in l] #On séléctionne les liens qui contiennent "all_comps/shooting"


# In[29]:


links #On re affiche les liens


# In[30]:


data = requests.get(f'http://fbref.com{links[0]}') #data devient une url valide en ajoutant la f string au debut


# In[31]:


shooting_stats = pd.read_html(data.text, match = "Shooting")[0] #


# In[32]:


shooting_stats


# In[33]:


shooting_stats.head()


# In[34]:


shooting_stats.columns = shooting_stats.columns.droplevel() #On enlève un niveau d'index (ici le niveau "For Manchester City"...)


# In[35]:


shooting_stats['Date'].head() #On affiche les 5 premières lignes de la colonne Date


# In[36]:


matches = matches[0]


# In[37]:


team_data = matches.merge(shooting_stats[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")


# Ici on fusionne 2 DF pandas. Pour ce faire, nous avons besoin d'une colonne en commun (clé), cette colonne qui est dans matches et shooting_stats sera utilisé pour aligner les lignes 

# In[38]:


team_data.head()


# In[39]:


years = list(range(2023,2019,-1)) #Création d'une liste contenant 2023 et 2022


# In[40]:


years


# In[41]:


all_matches = [] #Création d'un tableau vide 


# In[42]:


standings_url = 'https://fbref.com/en/comps/9/2022-2023/2022-2023-Premier-League-Stats' #L'url q`ue l'on va utiliser pour scrapper dans notre fonction


# In[43]:


import time
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
            team_name = team_url.split('/')[-1].replace('-Stats', '').replace('-', ' ') #On sépare le team_name de l'url 
            data = requests.get(team_url) #on fait une requete sur chaque lien d'équipe 
            matches = pd.read_html(data.text, match="Scores & Fixtures")[0] #On récupére le tableau qui match textuellement
            soup = BeautifulSoup(data.text) #On creer un objet beautifulSoup
            links = [l.get("href") for l in soup.find_all('a')] #On récupere tout les liens 
            links = [l for l in links if l and 'all_comps/shooting/' in l] #On récupére seulement les liens qui contiennet shooting
            data = requests.get(f"https://fbref.com{links[0]}") #on fait une requête get vers le premier lien (c'est tous les mêmes)
            shooting = pd.read_html(data.text, match="Shooting")[0] #Retourne un tableau de qui contient textuellement Shooting 
            shooting.columns = shooting.columns.droplevel() #Pour eviter d'avoir un multi index, on retire un index inutile
            try:
                team_data = matches.merge(shooting[["Date","Sh","SoT","Dist","FK","PK","PKatt"]], on="Date") #On essaye de fusionner les données à partir de "Date"
            except ValueError:
                continue #Si on a une erreur alors on continue
            team_data = team_data[team_data['Comp'] == 'Premier League'] #On ne veut travailler que sur la PL 
            team_data['Season'] = year #La saison est l'année (soit 2023 soit 2022)
            team_data['Team'] = team_name #La colonne Team va contenir le nom des équipes
            all_matches.append(team_data) #on ajoute notre tableau team_data dans all_matches
            time.sleep(4) #On met un sleep pour ne pas envoyer trop de requete au site et évité le blocage


# In[44]:


data


# In[45]:


match_df = pd.concat(all_matches) #On utilise concat quand on a les mêmes noms de colonne pour fusionner deux DF


# In[46]:


match_df.columns = [c.lower() for c in match_df.columns]
match_df.to_csv('matches1.csv')


# In[47]:


match_df.shape


# In[48]:


match_df


# In[49]:


38*20*4


# In[50]:


match_df['team'].value_counts()


# In[51]:


match_df.dtypes


# In[52]:


match_df['date'] = pd.to_datetime(match_df['date'])


# In[53]:


match_df.dtypes


# In[54]:


match_df.head(30)


# In[55]:


match_df.info()


# In[56]:


match_df['venue']


# In[57]:


match_df['venue_code'] = match_df['venue'].astype('category').cat.codes #On convertit pour pouvoir exploiter ces données (on ne veut pas de string)


# In[58]:


match_df['opp_code'] = match_df['opponent'].astype('category').cat.codes #On convertit pour pouvoir exploiter au mieux (on ne veut pas de string)


# In[59]:


match_df['opp_code'] #On regarde le opp_code


# In[60]:


match_df['hour'] = match_df['time'].replace(":.+","",regex=True).astype('int') #on créer une colonne hour, qui sera la colonne time en remplacant tout ce qu'il y a après ":" par rien en convertissant ce nombre en int


# In[61]:


match_df['hour'] #On affiche pour essayer


# In[62]:


match_df['day_code'] = match_df['date'].dt.dayofweek #dt est un accesseur pour les données de type datetime en pandas. C'est un attribut accessible via l'accèsseur .dt qui retourne le jour de la semaine sous forme d'un entier pour chaque élément de la série. Comme mentionné, le lundi est représenté par 0 et le dimanche par 6.


# In[63]:


match_df


# In[64]:


match_df['target'] = (match_df['result'] == 'W').astype('int') #Le resultat sera True ou False et on le convertit soit en 1 soit 0


# In[65]:


match_df.head()


# In[66]:


match_df.dtypes


# In[67]:


from sklearn.ensemble import RandomForestClassifier #On import notre classifier random forest (on peut dire que nous sommes dans un problème de classification)


# In[68]:


rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1) #n_estimators = nombre d'arbre dans la fôret, #min_samples_split = le minimum d'échantillon nécessaire pour diviser un arbre 


# In[69]:


train = match_df[match_df['date'] < '2023-01-01'] #Le Dataset d'entrainement est tout match_df avec les matchs d'avant 2023


# In[70]:


test = match_df[match_df['date'] > '2023-01-01'] #Le Dataset de test est tout match_df avec les matchs aprés 2023


# In[71]:


predictors = ['venue_code', 'opp_code', 'hour', 'day_code', 'xg', 'xga', 'poss', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt'] #predictors est le tableau 


# In[72]:


train


# In[73]:


means = train[predictors].mean()
train.loc[:, predictors] = train[predictors].fillna(means)


# In[74]:


rf.fit(train[predictors], train['target']) #En se basant sur les colonnes de predictors, on va essayer de prédire la colonne target


# In[75]:


preds = rf.predict(test[predictors]) #Avec notre classifier qui a appris les pattern grâce aux données d'entrainement, on lui passe cette fois les données de test qui DOIVENT être préparé de la même manière que les données d'entrainement.


# In[76]:


from sklearn.metrics import accuracy_score #Import de la méthode pour connaitre le score de précision TF/P+N


# In[77]:


accuracy_score(test['target'], preds) #Prend en argument la vraie étiquette et les étiquettes prédites et mesure la précision (TP/N+P)


# In[78]:


combined = pd.DataFrame(dict(actual = test['target'], prediction = preds)) #On créer une DF, les clés du dictionnaire deviennent les noms des colonnes


# In[79]:


pd.crosstab(index = combined['actual'], columns=combined['prediction']) #On creer un tableau avec comme index actual et comme colonne prediction. Le tabeau consiste à compter le nombre de fois ou les resultats correspondent


# In[80]:


from sklearn.metrics import precision_score #Importation de la méthode de calcul de precision du modéle


# In[81]:


precision_score(test['target'], preds) #vrai positif / prediction positive


# Let's try to improve our model ...

# In[82]:


grouped_matches = match_df.groupby('team') #On regroupe par nom d'équipe


# In[83]:


group = grouped_matches.get_group('Manchester City').sort_values('date') #


# In[84]:


group.head()


# In[85]:


def rolling_averages(group, cols, newcols):
    group = group.sort_values('date')
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=newcols)
    return group


# In[86]:


cols = ['gf', 'ga', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt']
new_cols = [f'{c}_rolling' for c in cols]


# In[87]:


new_cols


# In[88]:


rolling_averages(group, cols, new_cols)


# In[89]:


matches_rolling = match_df.groupby('team').apply(lambda x: rolling_averages(x, cols, new_cols))


# In[90]:


matches_rolling


# In[91]:


matches_rolling.dtypes


# In[92]:


matches_rolling.shape


# In[93]:


matches_rolling.index = range(matches_rolling.shape[0])


# In[94]:


from sklearn.impute import SimpleImputer
def make_prediction(data, predictors):
    train = data[data['date'] < '2023-01-01']
    test = data[data['date'] > '2023-01-01']
    columns_to_impute = ['xg', 'xga', 'poss', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt']
    # Créez un imputeur qui remplira les valeurs NaN par la moyenne des colonnes
    imputer = SimpleImputer(strategy='mean')
   # Imputer seulement sur les colonnes sélectionnées
    train_imputed = train.copy()
    test_imputed = test.copy()
    
    train_imputed[columns_to_impute] = imputer.fit_transform(train[columns_to_impute])
    test_imputed[columns_to_impute] = imputer.transform(test[columns_to_impute])
    # Entraînez le modèle avec les colonnes imputées et les autres prédicteurs
    rf.fit(train_imputed[predictors], train_imputed['target'])
    
    # Faites des prédictions sur l'ensemble de test
    preds = rf.predict(test_imputed[predictors])
    
    # Créez un DataFrame pour comparer les valeurs réelles et prédites
    combined = pd.DataFrame(dict(actual=test['target'], predicted=preds), index=test.index)
    
    # Calculez la précision des prédictions
    precision = precision_score(test["target"], preds)
    return combined, precision


# In[95]:


combined, precision = make_prediction(matches_rolling, predictors + new_cols)


# In[96]:


precision


# In[97]:


combined


# In[98]:


combined = combined.merge(matches_rolling[['date', 'team', 'opponent', 'result']], left_index=True,right_index=True) 


# In[99]:


combined


# In[100]:


matches_rolling[['date', 'team', 'opponent', 'result']]


# In[101]:


combined


# In[102]:


class MissingDict(dict):
    __missing__ = lambda self,key:key
    
map_values = {
    'Brighton and Hove Albion' : 'Brighton',
    'Manchester United' : 'Manchester Utd',
    'Newcastle United' : 'Newcastle Utd',
    'Tottenham Hotspur' : 'Tottenham',
    'West Ham United' : 'West Ham',
    'Wolverhampton Wanderers' : 'Wolves'
    }
mapping = MissingDict(**map_values)


# In[103]:


combined['new_team'] = combined['team'].map(mapping)


# In[104]:


combined


# In[105]:


merged = combined.merge(combined, left_on = ['date', 'new_team'], right_on = ['date', 'opponent'])


# In[106]:


merged.head()


# In[107]:


merged[(merged["predicted_x"]) == 1 & (merged["predicted_y"] == 0)]['actual_x'].value_counts()


# In[184]:


147/( 147 + 110)


# Poisson modèle using

# In[109]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.stats import poisson, skellam
import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[111]:


matches_rolling.head()


# In[113]:


matches_rolling['venue_int'] = matches_rolling['venue'].map({'Home' : 1, 'Away' : 0})


# In[115]:


matches_rolling


# In[144]:


goal_model_data = matches_rolling[['team', 'opponent','gf','venue_int']]


# In[154]:


goal_model_data = goal_model_data.copy()
goal_model_data.rename(columns={'gf': 'goals'}, inplace=True)


# In[155]:


goal_model_data.info()


# In[160]:


goal_model_data['team'] = goal_model_data['team'].astype('category')
goal_model_data['opponent'] = goal_model_data['opponent'].astype('category')
goal_model_data['venue_int'] = goal_model_data['venue_int'].astype(int)
goal_model_data['goals'] = goal_model_data['goals'].astype(float)  # ou int selon le contexte


# In[162]:


poisson_model = smf.glm(formula="goals ~ venue_int + team + opponent", data=goal_model_data, family=sm.families.Poisson()).fit()
print(poisson_model.summary())


# In[187]:


home_team = 'Manchester City'
away_team = 'Norwich City'


# In[188]:


home_score_rate = poisson_model.predict(pd.DataFrame(data = {'team' : home_team, 'opponent' : away_team, 'venue_int' : 1},index=[1]))
away_score_rate = poisson_model.predict(pd.DataFrame(data = {'team' : away_team, 'opponent' : home_team, 'venue_int' : 1},index=[1]))


# In[189]:


print(home_team + ' against ' + away_team + ' expect to score: ' + str(home_score_rate))
print(away_team + ' against ' + home_team + ' expect to score: ' + str(away_score_rate))


# In[190]:


def simulate_match(foot_model, homeTeam, awayTeam, max_goals=10):
    home_goals_avg = foot_model.predict(pd.DataFrame(data = {'team' : homeTeam, 'opponent' : awayTeam, 'venue_int' : 1},index=[1])).values[0]
    away_goals_avg = foot_model.predict(pd.DataFrame(data = {'team' : awayTeam, 'opponent' : homeTeam, 'venue_int' : 1},index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
    return (np.outer(np.array(team_pred[0]), np.array(team_pred[1])))


# In[191]:


max_goals = 5
score_matrix = simulate_match(poisson_model, home_team, away_team, max_goals)
score_matrix


# In[192]:


import seaborn as sns
ax = sns.heatmap(score_matrix, linewidth=0.7, annot=True)
ax.set_xlabel('Goals scored by ' + away_team)
ax.set_ylabel('Goals scored by ' + home_team)
plt.show()

