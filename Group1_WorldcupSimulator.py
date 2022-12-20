import numpy as np
import pandas as pd
from itertools import combinations
import warnings
from matplotlib import pyplot as plt
import seaborn as sns
from termcolor import colored, cprint

# from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# import tensorflow as tf

# from keras.utils import plot_model
from DNN import *
from trainers import *
warnings.simplefilter('ignore')
SEED = 101
SEPARATOR = colored(200*'=', 'red')

teams_worldcup = ['Qatar', 'Ecuador', 'Senegal', 'Netherlands', 'England', 'IR Iran', 'USA',
                  'Wales', 'Argentina', 'Saudi Arabia', 'Mexico', 'Poland', 'France', 
                  'Australia', 'Denmark', 'Tunisia', 'Spain', 'Costa Rica', 'Germany', 
                  'Japan', 'Belgium', 'Canada', 'Morocco', 'Croatia', 'Brazil', 'Serbia', 
                  'Switzerland', 'Cameroon', 'Portugal', 'Ghana', 'Uruguay', 'Korea Republic']


df = pd.read_csv('./Group1_updated_data.csv').drop('Unnamed: 0', axis=1)

def get_tournament_rank(tournament):
    if ('FIFA World Cup' in tournament or 'Confederations' in tournament):
        return 'International'
    elif (tournament == 'Friendly'):
        return tournament
    elif ('UEFA' in tournament):
        return 'Europe'
    elif ('African' in tournament):
        return 'Africa'
    elif ('AFC' in tournament):
        return 'Asia'
    elif ('CONCACAF' in tournament):
        return 'North America'
    elif ('Copa' in tournament):
        return 'South America'
    else :
        return 'Other'

columns_to_drop = ['home_team', 'away_team', 'home_team_score', 'away_team_score', 'date', 'tournament', 'city', 'country', 'neutral_location', 'home_win', 'home_draw', 'home_lose',
                   'tournament', 'tournament_rank', 'home_team_result', 
                   'home_team_continent', 'away_team_continent', 'year']
    

df['total_fifa_points_diff'] = df['home_team_total_fifa_points'] - df['away_team_total_fifa_points']
df['total_fifa_points_sum'] = df['home_team_total_fifa_points'] + df['away_team_total_fifa_points']
df['fifa_rank_diff'] = df['home_team_fifa_rank'] - df['away_team_fifa_rank']
df['fifa_rank_sum'] = df['home_team_fifa_rank'] + df['away_team_fifa_rank']
df['tournament_rank'] = df['tournament'].apply(get_tournament_rank)

# Encode categorical features
# 1. tournament rank
tournament_rank_encoder = OneHotEncoder(drop='first').fit(df['tournament_rank'].values.reshape(-1, 1))
tournament_rank = tournament_rank_encoder.transform(df['tournament_rank'].values.reshape(-1, 1)).toarray()
tournament_rank = pd.DataFrame(tournament_rank, columns=['tournament' + str(i) for i in range(tournament_rank.shape[1])])
df = pd.concat([df, tournament_rank], axis=1)

# 2. shoot_out
df['shoot_out'] = (df['shoot_out']=='Yes').astype(int)

# 3. continent
continent_encoder = OrdinalEncoder()
continent_encoder.fit(df['home_team_continent'].values.reshape(-1, 1))
df['home_team_continent_encoded'] = continent_encoder.transform(df['home_team_continent'].values.reshape(-1, 1))
df['away_team_continent_encoded'] = continent_encoder.transform(df['away_team_continent'].values.reshape(-1, 1))

# fill score's nulls by using 'bfill' and 'ffill'
df = df.fillna(method='bfill')
df = df.fillna(method='ffill')


X = df.drop(columns_to_drop, axis=1)
t = (df['home_team_result'] == 'Win').astype(int)
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2,  shuffle=True)

# change data types to np.array because it's data type is 'dataframe' and 'series'
X_train = np.array(X_train)
X_test = np.array(X_test)
t_train = np.array(t_train)
t_test = np.array(t_test)

# define the activation function
activation = 'elu'

# make DNN model with our own code (check it at DNN.py)
network = MultiLayerNet(34,[32,100,80,60,90,50,20,10],1)

# make trainer with our own code (check it at trainers.py)
trainer = Trainer(network, X_train, t_train, X_test, t_test)
trainer.train()

# worldcup_2022 team's data :  [rank, points, 4 OVAs]
teams_worldcup_2022 = {'Qatar':[50,1439.89,71,73,71,71], 'Ecuador': [44,1464.39,75,75,75,75], 'Senegal':[18,1584.38,78,79,76,77], 'Netherlands':[8,1694.51,83,82,82,83],
                'England':[5,1728.47,83,86,83,83], 'IR Iran':[20,1564.61,73,81,73,72], 'USA':[16,1627.48,77,77,78,76],'Wales':[19,1569.82,77,78,76,77],
                'Argentina':[3,1773.88,84,86,84,82], 'Saudi Arabia':[51,1437.78,71,71,72,71], 'Mexico':[13,1644.89,78,79,77,77], 'Poland':[26,1548.59,77,81,75,75],
                'France':[4,1759.78,83,86,82,82], 'Australia':[38,1488.72,72,72,72,71], 'Denmark':[10,1666.57,80,77,83,80], 'Tunisia':[30,1507.54,72,72,75,71],
                'Spain':[7,1715.22,83,83,85,83], 'Costa Rica':[31,1503.59,74,73,73,74], 'Germany':[11,1650.21,84,84,85,82], 'Japan':[24,1559.54,76,75,77,76],
                'Belgium':[2,1816.71,83,84,81,79], 'Canada':[41,1475,75,75,78,72], 'Morocco':[22,1563.55,77,79,73,78], 'Croatia':[12,1645.64,80,80,83,78], 
                'Brazil':[1,1841.3,85,85,85,83], 'Serbia':[21,1563.62,79,80,80,75], 'Switzerland':[15,1635.92,79,77,78,78], 'Cameroon':[43,1471.44,75,75,75,72], 
                'Portugal':[9,1676.56,83,83,83,84], 'Ghana':[61,1393,75,81,76,75], 'Uruguay':[14,1638.71,80,81,82,79], 'Korea Republic':[28,1530.3,76,79,74,75]}


# construct the test data for simulating
def get_team_data(team: str) -> list:

    #fifa_rank
    fifa_rank = teams_worldcup_2022[team][0]
    #fifa_point
    fifa_points = teams_worldcup_2022[team][1]
    
    #continent
    continent = df.loc[df['home_team']==team, 'home_team_continent_encoded'].mode()[0]

    # scores
    home_scores = df.loc[(df['home_team']==team).values & (df['away_team'].isin(teams_worldcup)).values].sort_values(by='date').drop('home_team_score', axis=1).filter(regex='score').filter(regex='home').iloc[:5].mean()
    away_scores = df.loc[(df['away_team']==team).values & (df['home_team'].isin(teams_worldcup)).values].sort_values(by='date').drop('away_team_score', axis=1).filter(regex='score').filter(regex='away').iloc[:5].mean()
    scores = (home_scores.values + away_scores.values) / 2
    
    # fifa_game stat
    ovas = teams_worldcup_2022[team][2:6]

    return [fifa_rank, fifa_points, continent] + [*scores] + [*ovas]


# define function which makes input data to fit with our model
def prepare_data(team1_data: list, team2_data: list, tournament_rank:str, shoot_out:bool) -> np.ndarray:
    fifa_pts_diff = team1_data[1] - team2_data[1]
    fifa_pts_sum = team1_data[1] + team2_data[1]
    fifa_rank_diff = team1_data[0] - team2_data[0]
    fifa_rank_sum = team1_data[0] + team2_data[0]
    tournaments = [*tournament_rank_encoder.transform(np.array(tournament_rank).reshape(-1, 1)).toarray()[0]]
    #X = [ ... ]
    X = np.array([team1_data[0], team2_data[0], team1_data[1], team2_data[1], shoot_out, team1_data[3], team2_data[3], *team1_data[4:], *team2_data[4:],
                 fifa_pts_diff, fifa_pts_sum, fifa_rank_diff, fifa_rank_sum, *tournaments, team1_data[2], team2_data[2]])


    return X.reshape(-1, 34)


# shape of X
""" X = [team1 rank, team2 rank, team1 fifa_pts, team2 fifa_pts, shoo_out(True,False),
        team1 GK/DEF/MID/ATT score , team2 GK/DEF/MID/ATT score , team1 TOT/DEF/MID/ATT overall, team2 TOT/DEF/MID/ATT overall,
        fifa pts diff, fifa pts sum, fifa rank diff, fifa rank sum, tournament one-hot-vector with 7 elements ,team1 continent, team2 continent] """


# define function which returns the prediction
def predict_match(team1, team2, shoot_out=0):
    team1 = get_team_data(team1)
    team2 = get_team_data(team2)
    inp1 = prepare_data(team1, team2, 'International', shoot_out=shoot_out)
    y_pred1 = network.predict(inp1)

    inp2 = prepare_data(team2, team1, 'International', shoot_out=shoot_out)
    y_pred2 = network.predict(inp2)

    y_pred = (1 + y_pred1 - y_pred2) / 2
    return y_pred


# make function to calculate the points at group stage
def calc_points(team1, team2):
    y_pred = predict_match(team1, team2)
    if (y_pred>=0.60):
        return 3, 0
    elif (y_pred<=0.40):
        return 0, 3
    else :
        return 1, 1

groups = np.array(teams_worldcup).reshape(8, 4)
groups_pts = []

print('\n\n\n','-'*10,'World Cup Start','-'*10)
for i, group in enumerate(groups):
    print(f'Group {i+1}: ', end=f"\n{20*'-'}\n")
    groups_pts.append(dict(zip(group, np.zeros(4))))
    matches = [*combinations(group, 2)]
    
    # play matches
    for team1, team2 in matches:
        pt1, pt2 = calc_points(team1, team2)
        print(f"{team1} : {pt1} VS {team2} : {pt2}")
        groups_pts[i][team1] += pt1
        groups_pts[i][team2] += pt2
    print(30*'=')


sorted_groups_pts = []

# define function for tie-breaking - just play one more match between two teams because we don't know their goal points
def draw_break(groups_pts):
    for i, group_pts in enumerate(groups_pts):
        group = dict(reversed(sorted(group_pts.items(), key=lambda item: item[1])))
        teams = [*group.keys()]
        points = [*group.values()]
        
        if points[0] <= points[1]:
            out = predict_match(teams[0], teams[1])
            if out < 0.5:
                temp = teams[0]
                teams[0] = teams[1]
                teams[1] = temp

        if points[1] <= points[2]:
            out = predict_match(teams[1], teams[2])
            if out < 0.5:
                temp = teams[1]
                teams[1] = teams[2]
                teams[2] = temp
                
        sorted_groups_pts.append(dict(zip(teams, points)))
    return sorted_groups_pts
    
sorted_groups_pts = draw_break(groups_pts)

sorted_teams = []
print(colored("Groups after group stage: \n", 'red'))
for group in sorted_groups_pts:
    print(group, end=f'\n{50*"-"}\n')
    teams = [*group.keys()]
    sorted_teams.append(teams[:2])
    


# the round of 16
to_16 = []
for i in range(0, 8, 2):
    to_16.append((sorted_teams[i][0], sorted_teams[i+1][1]))
    to_16.append((sorted_teams[i+1][0], sorted_teams[i][1]))
    
#make array to put winning teams of the round of 16
to_8 = []
print(colored(f'\n{50*"="}\nMatches results in the round of 16:\n{50*"-"}', 'red'))
for i, (team1, team2) in enumerate(to_16):
    out =  predict_match(team1, team2, shoot_out=1)
    winner = team1 if out > 0.5 else team2
    print(f'Match{i+1}: {team1} VS {team2}')
    print(f'Winner: {colored(winner, "green")}')
    to_8.append(winner)


# the round of 8
# to_8 = to_8[::2] + to_8[1::2]
to_8 = np.array(to_8[::2] + to_8[1::2]).reshape(-1, 2)
to_4 = []
print(colored(f'\n{50*"="}\nMatches results in the round of 8:\n{50*"-"}', 'blue'))
for i, (team1, team2) in enumerate(to_8):
    out =  predict_match(team1, team2, shoot_out=1)
    winner = team1 if out > 0.5 else team2
    print(f'Match{i+1}: {team1} VS {team2}')
    print(f'Winner: {colored(winner, "green")}')
    to_4.append(winner)


# the round of 4
to_4 = np.array(to_4).reshape(-1, 2)
to_final = []
to_third_place_playoff = []
print(colored(f'\n{50*"="}\nMatches results in the semi_fianl:\n{50*"-"}', 'yellow'))
for i, (team1, team2) in enumerate(to_4):
    out =  predict_match(team1, team2, shoot_out=1)
    winner = team1 if out > 0.5 else team2
    loser = team1 if out <= 0.5 else team2
    print(f'Match{i+1}: {team1} VS {team2}')
    print(f'Winner: {colored(winner, "green")}')
    to_final.append(winner)
    to_third_place_playoff.append(loser)


# third place play-off
third_out = predict_match(*to_third_place_playoff, shoot_out=1)
third = to_third_place_playoff[1 - int(third_out>0.5)]
fourth = to_third_place_playoff[int(third_out>0.5)]


# Final 
final_out = predict_match(*to_final, shoot_out=1)
Winner = to_final[1 - int(final_out>0.5)]
second = to_final[int(final_out>0.5)]

print(colored(f'\n{50*"="}\nMatches results in the final:\n{50*"-"}', 'yellow'))

#return the result 
print(f"Winner of the World Cup Qatar 2022: {colored(Winner, 'yellow')}")
print(f"Second place of the World Cup Qatar 2022: {colored(second, 'red')}")
print(f"Third place of the World Cup Qatar 2022: {colored(third, 'blue')}")
print(f"Fourth place of the World Cup Qatar 2022:", end=' ')
cprint(fourth, 'blue', 'on_grey')