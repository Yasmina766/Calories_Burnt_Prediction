import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

st.set_page_config(page_title='Calories Burn Prediction', layout='wide', page_icon='🔥')

st.markdown('<style>.stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }</style>', unsafe_allow_html=True)

@st.cache_data
def load_and_prep_data():
    calories = pd.read_csv('calories.csv')
    exercise = pd.read_csv('exercise.csv')
    df = pd.merge(exercise, calories, on='User_ID')
    df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})
    return df

try:
    df = load_and_prep_data()
    X = df.drop(columns=['User_ID', 'Calories'], axis=1)
    Y = df['Calories']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
except Exception as e:
    st.error(f'Data error: {e}')
    st.stop()

st.sidebar.header('📝 Input Parameters')
gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
age = st.sidebar.slider('Age', 10, 100, 25)
height = st.sidebar.slider('Height (cm)', 100, 250, 170)
weight = st.sidebar.slider('Weight (kg)', 30, 200, 70)
duration = st.sidebar.slider('Duration (min)', 1, 60, 15)
heart_rate = st.sidebar.slider('Heart Rate', 60, 150, 100)
body_temp = st.sidebar.slider('Body Temperature (C)', 35.0, 43.0, 39.0)

input_data = pd.DataFrame({
    'Gender': [0 if gender == 'Male' else 1],
    'Age': [age], 'Height': [height], 'Weight': [weight],
    'Duration': [duration], 'Heart_Rate': [heart_rate], 'Body_Temp': [body_temp]
})

st.title('🔥 Calories Burned Prediction Dashboard')
st.markdown('Compare algorithms to find the best prediction model.')

st.subheader('🛠️ Select Algorithm')
selected_algo = st.selectbox('Model:', ['XGBoost Regressor', 'Random Forest', 'Linear Regression', 'Lasso', 'Ridge'])

models = {
    'XGBoost Regressor': XGBRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(),
    'Ridge': Ridge()
}

if st.button('🚀 Run Prediction'):
    model = models[selected_algo]
    model.fit(X_train, Y_train)
    test_pred = model.predict(X_test)
    mae = metrics.mean_absolute_error(Y_test, test_pred)
    r2 = metrics.r2_score(Y_test, test_pred)
    prediction = model.predict(input_data)
    
    st.success(f'Model trained using {selected_algo}!')
    c1, c2, c3 = st.columns(3)
    c1.metric('Predicted Calories', f'{prediction[0]:.2f} kcal')
    c2.metric('Error (MAE)', f'{mae:.3f}')
    c3.metric('Accuracy (R2)', f'{r2:.4f}')
    st.info(f'**Note for Instructor:** {selected_algo} shows an accuracy score of {r2:.4f}.')
