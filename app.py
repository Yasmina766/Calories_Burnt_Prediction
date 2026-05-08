import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CalorieIQ — Burn Predictor",
    page_icon="🔥",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Hero banner */
.hero {
    text-align: center;
    padding: 2rem 1rem 1rem;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f093fb, #f5576c, #fda085);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #a0aec0;
    font-size: 1.1rem;
    font-weight: 300;
}

/* Glass card */
.glass-card {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
}

/* Section label */
.section-label {
    color: #f093fb;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* Custom input labels */
.stTextInput label, .stSelectbox label, .stSlider label {
    color: #e2e8f0 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

/* Input boxes */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-size: 1rem !important;
    padding: 0.6rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #f093fb !important;
    box-shadow: 0 0 0 2px rgba(240,147,251,0.25) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 12px !important;
    color: #fff !important;
}

/* Sliders */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #f093fb, #f5576c) !important;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #f093fb, #f5576c) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.8rem 2rem !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(245, 87, 108, 0.4) !important;
    letter-spacing: 0.05em !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(245, 87, 108, 0.6) !important;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, rgba(240,147,251,0.15), rgba(245,87,108,0.15));
    border: 1px solid rgba(240,147,251,0.4);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-cal {
    font-size: 4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f093fb, #f5576c, #fda085);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.result-unit {
    color: #a0aec0;
    font-size: 1rem;
    margin-top: 0.3rem;
}
.result-label {
    color: #e2e8f0;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* Algorithm comparison */
.algo-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.algo-name {
    color: #cbd5e0;
    font-size: 0.88rem;
    font-weight: 500;
}
.algo-val {
    font-size: 0.95rem;
    font-weight: 700;
}

/* Best algo badge */
.best-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(104,211,145,0.2), rgba(56,178,172,0.2));
    border: 1px solid rgba(104,211,145,0.5);
    color: #68d391;
    border-radius: 20px;
    padding: 0.15rem 0.6rem;
    font-size: 0.7rem;
    font-weight: 700;
    margin-left: 0.5rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Best algo info card */
.best-algo-card {
    background: linear-gradient(135deg, rgba(104,211,145,0.1), rgba(56,178,172,0.1));
    border: 1px solid rgba(104,211,145,0.3);
    border-radius: 16px;
    padding: 1rem 1.5rem;
    margin-top: 1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.best-algo-icon {
    font-size: 2rem;
}
.best-algo-text h4 {
    color: #68d391 !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0 !important;
}
.best-algo-text p {
    color: #e2e8f0;
    font-size: 1rem;
    font-weight: 700;
    margin: 0.2rem 0 0;
}
.best-algo-text small {
    color: #a0aec0;
    font-size: 0.78rem;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.1);
    margin: 1.5rem 0;
}

/* Info badges */
.badge {
    display: inline-block;
    background: rgba(240,147,251,0.15);
    border: 1px solid rgba(240,147,251,0.3);
    color: #f093fb;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.2rem;
}

/* Column headers */
h3 {
    color: #e2e8f0 !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Data & Model Loading ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models on dataset…")
def load_models():
    df = pd.read_csv("calories.csv")

    df.replace({"male": 0, "female": 1}, inplace=True)

    # Drop Weight (not entered by user), keep Duration and all other features
    df.drop(["Weight"], axis=1, inplace=True)

    features = df.drop(["User_ID", "Calories"], axis=1)
    target   = df["Calories"].values

    X_train, X_val, Y_train, Y_val = train_test_split(
        features, target, test_size=0.1, random_state=22
    )

    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)

    model_dict = {
        "Linear Regression": LinearRegression(),
        "XGBoost":           XGBRegressor(verbosity=0),
        "Lasso":             Lasso(),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
        "Ridge":             Ridge(),
    }

    # Train and evaluate each model → pick best by MAE on validation set
    mae_scores = {}
    for name, m in model_dict.items():
        m.fit(X_train_sc, Y_train)
        preds = m.predict(X_val_sc)
        mae_scores[name] = round(mean_absolute_error(Y_val, preds), 3)

    best_algo = min(mae_scores, key=mae_scores.get)

    # Feature order: Gender, Age, Height, Heart_Rate, Body_Temp, Duration
    feature_cols = list(features.columns)

    return model_dict, scaler, mae_scores, best_algo, feature_cols


models, scaler, mae_scores, best_algo, feature_cols = load_models()


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔥 CalorieIQ</h1>
    <p>Predict how many calories you'll burn — powered by 5 ML algorithms</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-bottom:1.5rem;">
    <span class="badge">Linear Regression</span>
    <span class="badge">XGBoost</span>
    <span class="badge">Lasso</span>
    <span class="badge">Random Forest</span>
    <span class="badge">Ridge</span>
</div>
""", unsafe_allow_html=True)

# ── Best Algorithm Banner ─────────────────────────────────────────────────────
algo_colors = {
    "Linear Regression": "#63b3ed",
    "XGBoost":           "#f6ad55",
    "Lasso":             "#68d391",
    "Random Forest":     "#fc8181",
    "Ridge":             "#b794f4",
}
best_color = algo_colors[best_algo]
best_mae   = mae_scores[best_algo]

st.markdown(f"""
<div class="best-algo-card">
    <div class="best-algo-icon">🏆</div>
    <div class="best-algo-text">
        <h4>Best Performing Algorithm (on validation set)</h4>
        <p style="color:{best_color};">{best_algo}</p>
        <small>Lowest MAE: <strong style="color:{best_color};">{best_mae} kcal</strong> — most accurate on unseen data</small>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">📝 Your Profile & Exercise Data</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"], help="Biological sex")

    age_input = st.text_input("Age (years)", value="25", help="Type your age")
    try:
        age = int(age_input)
        if not (1 <= age <= 120):
            st.warning("Age should be between 1 and 120.")
            age = 25
    except ValueError:
        st.error("Please enter a valid number for Age.")
        age = 25

    height_input = st.text_input("Height (cm)", value="170", help="Type your height in cm")
    try:
        height = float(height_input)
        if not (100 <= height <= 250):
            st.warning("Height should be between 100 and 250 cm.")
            height = 170.0
    except ValueError:
        st.error("Please enter a valid number for Height.")
        height = 170.0

with col2:
    heart_rate = st.slider(
        "Heart Rate (bpm)",
        min_value=60, max_value=180, value=100, step=1,
        help="Average heart rate during exercise"
    )

    body_temp = st.slider(
        "Body Temperature (°C)",
        min_value=37.0, max_value=42.0, value=40.0, step=0.1,
        format="%.1f",
        help="Body temp during exercise"
    )

    duration = st.slider(
        "Exercise Duration (minutes)",
        min_value=1, max_value=120, value=30, step=1,
        help="How long you exercised"
    )

    algo_choice = st.selectbox(
        "Primary Algorithm",
        list(models.keys()),
        index=list(models.keys()).index(best_algo),   # default = best
        help="Algorithm to highlight in the result"
    )

st.markdown('</div>', unsafe_allow_html=True)


# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("⚡ Predict Calories Burned"):

    gender_val = 0 if gender == "Male" else 1

    # Build input in the same column order as training data
    # feature_cols order is whatever came from df.drop(["User_ID","Calories"])
    input_map = {
        "Gender":     gender_val,
        "Age":        age,
        "Height":     height,
        "Heart_Rate": heart_rate,
        "Body_Temp":  body_temp,
        "Duration":   duration,
    }
    input_arr    = np.array([[input_map[c] for c in feature_cols]])
    input_scaled = scaler.transform(input_arr)

    predictions = {}
    for name, model in models.items():
        pred = model.predict(input_scaled)[0]
        predictions[name] = max(0.0, round(pred, 1))

    primary_pred = predictions[algo_choice]

    # ── Result display ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Estimated Calories Burned</div>
        <div class="result-cal">{primary_pred}</div>
        <div class="result-unit">kcal &nbsp;·&nbsp; {algo_choice}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── All algorithms comparison ────────────────────────────────────────────
    st.markdown('<div class="glass-card" style="margin-top:1.5rem;">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📊 All Algorithm Predictions</div>', unsafe_allow_html=True)

    # Sort by prediction value descending for nicer display
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    for name, val in sorted_preds:
        is_selected = name == algo_choice
        is_best     = name == best_algo
        highlight   = "font-weight:800;" if is_selected else ""
        star        = "★ " if is_selected else ""
        best_tag    = '<span class="best-badge">🏆 Best</span>' if is_best else ""
        mae_tag     = f'<span style="color:#718096;font-size:0.75rem;margin-left:0.5rem;">MAE {mae_scores[name]}</span>'

        st.markdown(f"""
        <div class="algo-row">
            <span class="algo-name">{star}{name}{best_tag}{mae_tag}</span>
            <span class="algo-val" style="color:{algo_colors[name]}; {highlight}">{val} kcal</span>
        </div>
        """, unsafe_allow_html=True)

    avg_pred = round(np.mean(list(predictions.values())), 1)
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;padding:0.8rem 0 0;">
        <span style="color:#f093fb;font-weight:700;font-size:0.9rem;">🔥 Ensemble Average</span>
        <span style="color:#f093fb;font-weight:800;font-size:1.05rem;">{avg_pred} kcal</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Input summary ────────────────────────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🧾 Input Summary</div>', unsafe_allow_html=True)
    summary_items = {
        "Gender":   gender,
        "Age":      f"{age} yrs",
        "Height":   f"{height} cm",
        "Heart Rate": f"{heart_rate} bpm",
        "Body Temp": f"{body_temp:.1f} °C",
        "Duration": f"{duration} min",
    }
    cols = st.columns(len(summary_items))
    for col, (k, v) in zip(cols, summary_items.items()):
        col.markdown(f"""
        <div style="text-align:center;">
            <div style="color:#a0aec0;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;">{k}</div>
            <div style="color:#e2e8f0;font-size:1rem;font-weight:700;margin-top:0.2rem;">{v}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#4a5568;font-size:0.78rem;margin-top:2rem;padding-bottom:1rem;">
    Built with Streamlit · Scikit-learn · XGBoost &nbsp;|&nbsp; 15,000 samples dataset
</div>
""", unsafe_allow_html=True)
