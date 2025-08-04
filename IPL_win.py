# import pandas as pd
# match=pd.read_csv('C:/Users/abhay/IPL_dataset_2008_2019/matches.csv')
# print(match.head())

# # %%writefile read_csv_example.py
# import pandas as pd

# df = pd.read_csv("C:/Users/abhay/IPL_dataset_2008_2019/matches.csv")  # replace with your file name
# print(df.head())
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns',None)

match=pd.read_csv('C:/Users/abhay/IPL_dataset_2008_2019/matches.csv')
delivery=pd.read_csv('C:/Users/abhay/IPL_dataset_2008_2019/deliveries.csv')

print(match.head())
print(delivery.head())

# print(match.info())

total_score_df=delivery.groupby(['match_id','inning'])['total_runs'].sum().reset_index()
total_score_df['total_runs']=total_score_df['total_runs'].apply(lambda x:x+1)#add 1 on target value

# print(total_score_df)

total_score_df=total_score_df[total_score_df['inning']==1]
# total_score_df
# delivery[delivery['match_id']==7]
match_df=match.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')
match_df['team1'].unique()
match_df['team1']=match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2']=match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team1']=match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2']=match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team1'].unique()
teams=['Sunrisers Hyderabad', 'Mumbai Indians','Royal Challengers Bangalore',
       'Kolkata Knight Riders', 'Delhi Capitals', 'Kings XI Punjab',
       'Chennai Super Kings', 'Rajasthan Royals']
match_df=match_df[match_df['team1'].isin(teams)]
match_df=match_df[match_df['team2'].isin(teams)]
# match_df.shape
match_df=match_df[match_df['dl_applied']==0]

match_sort_df=match_df[['match_id','city','winner','total_runs']]
delivery_df=match_sort_df.merge(delivery,on='match_id')
delivery_df

delivery_df=delivery_df[delivery_df['inning']==2]
delivery_df['current_score']=delivery_df.groupby('match_id')['total_runs_y'].cumsum()
# print(delivery_df)
delivery_df['runs_left']=delivery_df['total_runs_x']-delivery_df['current_score']
delivery_df['balls_left']=126-(delivery_df['over']*6+delivery_df['ball'])

delivery_df['player_dismissed']=delivery_df['player_dismissed'].fillna("0")
delivery_df['wicket']=delivery_df['player_dismissed'].apply(lambda x:0 if x=="0" else 1)
delivery_df['wicket']=delivery_df.groupby('match_id')['wicket'].cumsum()
delivery_df['wicket_left']=10-delivery_df['wicket']
# '''
# import streamlit as st
# import pandas as pd
# import numpy as np

# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.ensemble import RandomForestClassifier
# pd.set_option('display.max_columns',None)
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

match=pd.read_csv('C:/Users/abhay/csv/Gen_Final_df_of_Cricket.csv')
st.title('IPL Win Predictor')


teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

# ----------------- Model Pipeline -----------------
# Define preprocessing transformer
trf = ColumnTransformer(
    [('trf', OneHotEncoder(sparse_output=False, drop='first'),
      ['batting_team', 'bowling_team', 'city'])],
    remainder='passthrough'
)

# Define pipeline
pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', RandomForestClassifier())
])

X_train = match.drop(columns='result')
y_train = match['result']

pipe.fit(X_train, y_train)

# ----------------- User Input -----------------
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target')

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets out')

# ----------------- Prediction -----------------
if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    
    # input_df = pd.DataFrame({
    #     'batting_team': [batting_team],
    #     'bowling_team': [bowling_team],
    #     'city': [selected_city],
    #     'runs_left': [runs_left],
    #     'balls_left': [balls_left],
    #     'wicket_left': [wickets_left],
    #     'total_runs_x': [target],
    #     'crr': [crr],
    #     'rr': [rrr]
    # })
    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':               [balls_left],'wicket_left':[wickets],'total_runs_x':[target],'crr':[crr],'rr':[rrr]})


    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    st.header(f"{batting_team} - {round(win * 100)}% chance to win")
    st.header(f"{bowling_team} - {round(loss * 100)}% chance to win")
