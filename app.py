import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# --- Page Config ---
st.set_page_config(page_title="Calories Burn Prediction", layout="wide", page_icon="🔥")

# --- Data Loading ---
@st.cache_data
def load_and_prep_data():
    try:
        calories = pd.read_csv('calories.csv')
        exercise = pd.read_csv('exercise.csv')
        df = pd.merge(exercise, calories, on='User_ID')
        df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})
        return df
    except Exception as e:
        return None

df = load_and_prep_data()

if df is not None:
    X = df.drop(columns=['User_ID', 'Calories'])
    Y = df['Calories']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    # --- Sidebar ---
    st.sidebar.header("📝 Input Parameters")
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    age = st.sidebar.slider("Age", 10, 100, 25)
    height = st.sidebar.slider("Height (cm)", 100, 250, 170)
    weight = st.sidebar.slider("Weight (kg)", 30, 200, 70)
    duration = st.sidebar.slider("Duration (min)", 1, 60, 15)
    heart_rate = st.sidebar.slider("Heart Rate", 60, 150, 100)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 35.0, 43.0, 39.0)

    input_data = pd.DataFrame({
        'Gender': [0 if gender == "Male" else 1],
        'Age': [age], 'Height': [height], 'Weight': [weight],
        'Duration': [duration], 'Heart_Rate': [heart_rate], 'Body_Temp': [body_temp]
    })

    # --- Main ---
    st.title("🔥 Calories Burned Prediction Dashboard")
    st.markdown("Compare machine learning models to see which one is most accurate.")

    st.subheader("🛠️ Step 1: Select Algorithm")
    selected_algo = st.selectbox("Choose a model:", ["XGBoost Regressor", "Random Forest", "Linear Regression", "Lasso", "Ridge"])

    models = {
        "XGBoost Regressor": XGBRegressor(),
        "Random Forest": RandomForestRegressor(n_estimators=100),
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge()
    }

    if st.button("🚀 Run Algorithm"):
        with st.spinner('Calculating...'):
            model = models[selected_algo]
            model.fit(X_train, Y_train)
            test_pred = model.predict(X_test)
            mae = metrics.mean_absolute_error(Y_test, test_pred)
            r2 = metrics.r2_score(Y_test, test_pred)
            prediction = model.predict(input_data)
            
            st.success(f"Algorithm: {selected_algo}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Calories", f"{prediction[0]:.2f} kcal")
            col2.metric("Mean Absolute Error", f"{mae:.3f}")
            col3.metric("R² Score (Accuracy)", f"{r2:.4f}")
            st.info(f"**Note for TA:** Higher R² and lower MAE indicate a better model. Usually, XGBoost wins.")
else:
    st.error("Make sure 'calories.csv' and 'exercise.csv' are in the same folder as this script.")
