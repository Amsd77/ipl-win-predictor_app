import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# import seaborn as sns
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
# pd.set_option('display.max_columns',None)

match=pd.read_csv('Gen_Final_df_of_Cricket.csv')
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
    ('step2', RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train = match.drop(columns='result')
y_train = match['result']

pipe.fit(X_train, y_train)

def simulate_win_curve(pipe, batting_team, bowling_team, city, target, score, overs, wickets):
    """
    Simulates win probabilities over the remaining overs with projected scoring.
    """
    batting_probs = []
    bowling_probs = []
    overs_list = []

    wickets_left = 10 - wickets
    current_score = score

    # Average run rate so far
    avg_run_rate = current_score / overs if overs > 0 else 0

    # Loop over overs till 20
    for o in range(int(overs), 20):
        for b in range(6):
            curr_over = o + b/6
            balls_left = 120 - int(curr_over * 6)

            # Project runs at current run rate
            projected_score = current_score + avg_run_rate * (curr_over - overs)
            runs_left = max(target - projected_score, 0)

            # Current run rate
            crr = projected_score / curr_over if curr_over > 0 else 0

            # Required run rate
            rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

            # Input for model
            input_df = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [city],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wicket_left': [wickets_left],
                'total_runs_x': [target],
                'crr': [crr],
                'rr': [rrr]
            })

            # Get probabilities
            probs = pipe.predict_proba(input_df)[0]
            batting_probs.append(probs[1])  # batting win prob
            bowling_probs.append(probs[0])  # bowling win prob
            overs_list.append(curr_over)

    return overs_list, batting_probs, bowling_probs


# ----------------- User Input -----------------
with st.form("match_form"):
    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox('Select the batting team', sorted(teams), key="batting_team")
    with col2:
        bowling_team = st.selectbox('Select the bowling team', sorted(teams), key="bowling_team")

    selected_city = st.selectbox('Select host city', sorted(cities), key="city")

    target = st.number_input('Target', key="target")

    col3, col4, col5 = st.columns(3)
    with col3:
        score = st.number_input('Score', key="score")
    with col4:
        overs = st.number_input('Overs completed', key="overs")
    with col5:
        wickets = st.number_input('Wickets out', key="wickets")

    # Submit button for form
    # submitted = st.form_submit_button("Predict Probability")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        submitted = st.form_submit_button("Predict Probability")
    with col_btn2:
        reset = st.form_submit_button("🔄 Reset / Clear")

if reset:
    for key in ["batting_team", "bowling_team", "city", "target", "score", "overs", "wickets"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


comment_1='''
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
    '''

# ----------------- Prediction -----------------
if submitted:
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team':[batting_team],
        'bowling_team':[bowling_team],
        'city':[selected_city],
        'runs_left':[runs_left],
        'balls_left':[balls_left],
        'wicket_left':[wickets_left],
        'total_runs_x':[target],
        'crr':[crr],
        'rr':[rrr]
    })

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    st.success(f"{batting_team}: {win*100:.2f}% | {bowling_team}: {loss*100:.2f}%")

    # 📊 Probability Curve

    x, batting_probs, bowling_probs = simulate_win_curve(
    pipe, batting_team, bowling_team, selected_city, target, score, overs, wickets
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, [p*100 for p in batting_probs], label=f"{batting_team} win %", color="blue")
    ax.plot(x, [p*100 for p in bowling_probs], label=f"{bowling_team} win %", color="orange")
    ax.axhline(50, color="red", linestyle="--", label="50% reference")
    ax.set_xlabel("Overs")
    ax.set_ylabel("Win Probability (%)")
    ax.set_title("Win Probability Progression")
    ax.legend()
    st.pyplot(fig)
