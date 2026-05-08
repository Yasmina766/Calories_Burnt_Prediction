import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Calories Predictor", page_icon="🔥", layout="centered")

# ── Load & train ──────────────────────────────────────────
@st.cache_resource(show_spinner="جاري تدريب النماذج...")
def load_models():
    df = pd.read_csv("calories.csv")
    df.replace({"male": 0, "female": 1}, inplace=True)
    df.drop(["Weight"], axis=1, inplace=True)

    X = df.drop(["User_ID", "Calories"], axis=1)
    y = df["Calories"].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=22)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    models = {
        "Linear Regression": LinearRegression(),
        "XGBoost":           XGBRegressor(verbosity=0),
        "Lasso":             Lasso(),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
        "Ridge":             Ridge(),
    }

    mae_scores = {}
    for name, m in models.items():
        m.fit(X_train_s, y_train)
        mae_scores[name] = round(mean_absolute_error(y_val, m.predict(X_val_s)), 2)

    best = min(mae_scores, key=mae_scores.get)
    return models, scaler, mae_scores, best, list(X.columns)

models, scaler, mae_scores, best_algo, feature_cols = load_models()

# ── UI ────────────────────────────────────────────────────
st.title("🔥 Calories Burnt Predictor")
st.caption("ادخل بياناتك واضغط Predict")
st.divider()

col1, col2 = st.columns(2)

with col1:
    gender   = st.selectbox("الجنس", ["Male", "Female"])
    age      = st.number_input("العمر", min_value=1, max_value=120, value=25)
    height   = st.number_input("الطول (cm)", min_value=100, max_value=250, value=170)

with col2:
    duration   = st.slider("مدة التمرين (دقيقة)", 1, 120, 30)
    heart_rate = st.slider("معدل ضربات القلب", 60, 180, 100)
    body_temp  = st.slider("حرارة الجسم (°C)", 37.0, 42.0, 40.0, step=0.1)

st.divider()

if st.button("⚡ Predict", use_container_width=True, type="primary"):
    inp = {
        "Gender": 0 if gender == "Male" else 1,
        "Age": age, "Height": height,
        "Duration": duration, "Heart_Rate": heart_rate, "Body_Temp": body_temp,
    }
    arr    = np.array([[inp[c] for c in feature_cols]])
    scaled = scaler.transform(arr)

    preds = {name: max(0.0, round(m.predict(scaled)[0], 1)) for name, m in models.items()}
    best_pred = preds[best_algo]

    # Result
    st.success(f"**{best_pred} kcal** — بناءً على أدق نموذج ({best_algo})")

    # Comparison table
    st.write("##### مقارنة النماذج")
    rows = [{"النموذج": n, "التنبؤ (kcal)": v, "MAE": mae_scores[n]} for n, v in preds.items()]
    df_res = pd.DataFrame(rows).set_index("النموذج")
    st.dataframe(df_res, use_container_width=True)
