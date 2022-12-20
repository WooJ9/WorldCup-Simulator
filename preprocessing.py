import pandas as pd
from tqdm import tqdm 
import math
df1 = pd.read_csv('./fifaova.csv') #, encoding='CP949'
df2 = pd.read_csv('./matches.csv', encoding='CP949')


pbar = tqdm(df2.iterrows(), total = len(df2))

for index2, row2 in pbar:
    for index1, row1 in df1.iterrows():
        if row1.YEAR == row2.year and row1.TEAM == row2.home_team:
            df2.at[index2, 'home_OVA'] = row1.OVA
            df2.at[index2, 'home_ATT'] = row1.ATT
            df2.at[index2, 'home_MID'] = row1.MID
            df2.at[index2, 'home_DEF'] = row1.DEF
        elif math.isnan(df2.at[index2, 'home_OVA']):
            df2.at[index2, 'home_OVA'] = 85 - (row2.home_team_fifa_rank)*35/211
            df2.at[index2, 'home_ATT'] = 85 - (row2.home_team_fifa_rank)*35/211
            df2.at[index2, 'home_MID'] = 85 - (row2.home_team_fifa_rank)*35/211
            df2.at[index2, 'home_DEF'] = 85 - (row2.home_team_fifa_rank)*35/211
        if row1.YEAR == row2.year and row1.TEAM == row2.away_team:
            df2.at[index2, 'away_OVA'] = row1.OVA
            df2.at[index2, 'away_ATT'] = row1.ATT
            df2.at[index2, 'away_MID'] = row1.MID
            df2.at[index2, 'away_DEF'] = row1.DEF
        elif math.isnan(df2.at[index2, 'away_OVA']):
            df2.at[index2, 'away_OVA'] = 85 - (row2.away_team_fifa_rank)*35/211
            df2.at[index2, 'away_ATT'] = 85 - (row2.away_team_fifa_rank)*35/211
            df2.at[index2, 'away_MID'] = 85 - (row2.away_team_fifa_rank)*35/211
            df2.at[index2, 'away_DEF'] = 85 - (row2.away_team_fifa_rank)*35/211


df2.to_csv("./Group1_updated_data.csv",encoding="utf-8-sig")