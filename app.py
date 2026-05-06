import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Calories Burnt Predictor",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background: linear-gradient(135deg, #fbeaf0 0%, #f4c0d1 40%, #fbeaf0 100%) !important;
    color: #4b1528;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; padding-bottom: 2rem; max-width: 780px; }

/* Sliders */
.stSlider > div > div > div       { background: #d4537e !important; }
.stSlider > div > div > div > div { background: #d4537e !important; }

/* Selectbox */
.stSelectbox > div > div {
    background: white !important;
    border: 1px solid #f4c0d1 !important;
    border-radius: 10px !important;
}

/* Radio pills */
div[role="radiogroup"] { gap: 8px !important; flex-wrap: wrap; }
div[role="radiogroup"] label {
    background: white !important;
    border: 1px solid #f4c0d1 !important;
    border-radius: 20px !important;
    padding: 4px 14px !important;
    color: #993556 !important;
    font-size: 0.85rem !important;
}
div[role="radiogroup"] label:has(input:checked) {
    background: #d4537e !important;
    color: white !important;
    border-color: #d4537e !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #d4537e, #993556) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.7rem 2rem !important;
    width: 100%;
    box-shadow: 0 4px 16px rgba(212,83,126,0.35) !important;
    transition: all 0.2s;
}
.stButton > button:hover { transform: translateY(-2px); }

/* Section label */
.sec-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #993556;
    margin-bottom: 0.8rem;
    border-left: 3px solid #d4537e;
    padding-left: 8px;
}

/* Result card */
.result-wrap {
    background: linear-gradient(135deg, #fbeaf0, #f4c0d1);
    border: 1px solid #ed93b1;
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
}
.result-num  { font-size: 5rem; font-weight: 600; color: #72243e; line-height: 1; }
.result-unit { font-size: 0.9rem; color: #993556; font-family: 'DM Mono', monospace; margin-top: 4px; }
.result-algo { font-size: 0.75rem; margin-top: 10px; background: white;
               display: inline-block; padding: 3px 12px; border-radius: 20px;
               border: 1px solid #f4c0d1; color: #4b1528; }

/* Metric cards */
.metric-card {
    background: white;
    border: 1px solid #f4c0d1;
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
}
.metric-val { font-size: 1.6rem; font-weight: 600; color: #72243e; }
.metric-lbl { font-size: 0.7rem; color: #993556; font-family: 'DM Mono', monospace;
              letter-spacing: 1px; text-transform: uppercase; margin-top: 4px; }

/* Context rows */
.ctx-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #fbeaf0;
    font-size: 0.85rem;
}

/* Performance table - keep streamlit default dark-ish but override header */
.stDataFrame thead tr th {
    background: #fbeaf0 !important;
    color: #4b1528 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Train models ───────────────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    exercise = pd.read_csv("exercise.csv")
    calories = pd.read_csv("calories.csv")
    df = exercise.merge(calories, on="User_ID")

    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])

    features = df[["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]]
    target   = df["Calories"].values

    X_train, X_val, Y_train, Y_val = train_test_split(
        features, target, test_size=0.1, random_state=22
    )
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    model_defs = {
        "XGBoost":           XGBRegressor(random_state=42),
        "Random Forest":     RandomForestRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
        "Lasso":             Lasso(),
        "Ridge":             Ridge(),
    }
    fitted, perf = {}, {}
    for name, mdl in model_defs.items():
        mdl.fit(X_train_s, Y_train)
        perf[name] = {
            "train_mae": round(mean_absolute_error(Y_train, mdl.predict(X_train_s)), 2),
            "val_mae":   round(mean_absolute_error(Y_val,   mdl.predict(X_val_s)),   2),
        }
        fitted[name] = mdl

    return fitted, perf, scaler


models, perf, scaler = load_and_train()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-bottom:1.8rem;">
  <div style="font-size:2.4rem; font-weight:600; color:#4b1528; letter-spacing:-1px;">
    🌸 Calories Burnt Predictor
  </div>
  <div style="font-size:0.85rem; color:#993556; font-family:'DM Mono',monospace; margin-top:6px;">
    Enter your data · choose an algorithm · press predict
  </div>
</div>
""", unsafe_allow_html=True)

# ── Input form ─────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">Your Profile & Exercise Data</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    gender     = st.selectbox("Gender", ["Male", "Female"])
    height     = st.slider("Height (cm)", 140, 210, 170)
    weight     = st.slider("Weight (kg)", 40, 150, 70)
    duration   = st.slider("Duration (min)", 1, 120, 30)
with col2:
    age        = st.slider("Age (years)", 15, 80, 30)
    heart_rate = st.slider("Heart Rate (bpm)", 60, 200, 100)
    body_temp  = st.slider("Body Temperature (°C)", 36.0, 42.0, 39.0, step=0.1)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="sec-label">Choose Algorithm</div>', unsafe_allow_html=True)
selected = st.radio("", list(models.keys()), horizontal=True, label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🌸  Predict Calories Burned")

# ── Only show results after button click ───────────────────────────────────────
if predict_clicked:

    gender_enc  = 1 if gender == "Male" else 0
    user_input  = np.array([[gender_enc, age, height, weight, duration, heart_rate, body_temp]])
    user_scaled = scaler.transform(user_input)

    prediction = max(0.0, float(models[selected].predict(user_scaled)[0]))
    all_preds  = {n: max(0.0, float(m.predict(user_scaled)[0])) for n, m in models.items()}

    # ── Result card ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-wrap">
        <div style="font-size:0.72rem;color:#993556;font-family:'DM Mono',monospace;
                    letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">
            Estimated Burn
        </div>
        <div class="result-num">{prediction:.0f}</div>
        <div class="result-unit">kilocalories</div>
        <div class="result-algo">via {selected}</div>
    </div>
    """, unsafe_allow_html=True)

    # BMI
    bmi     = weight / ((height / 100) ** 2)
    bmi_cat = ("Underweight" if bmi < 18.5 else
                "Normal weight" if bmi < 25 else
                "Overweight"   if bmi < 30 else "Obese")
    bmi_col = "#3b6d11" if bmi < 25 else "#ba7517" if bmi < 30 else "#a32d2d"
    bmi_bg  = "#eaf3de" if bmi < 25 else "#faeeda" if bmi < 30 else "#fcebeb"
    st.markdown(f"""
    <div style="text-align:center;margin-top:10px;font-size:0.85rem;color:#72243e;">
        BMI: <strong>{bmi:.1f}</strong>
        <span style="background:{bmi_bg};color:{bmi_col};padding:2px 10px;
                     border-radius:20px;font-size:0.75rem;margin-left:6px;">
            {bmi_cat}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Caloric context ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Caloric Context</div>', unsafe_allow_html=True)
    benchmarks = [
        ("🍎", "Apple", 95),
        ("🍕", "Pizza slice", 285),
        ("🍔", "Hamburger", 550),
        ("🏃", "30-min run avg", 300),
        ("🥤", "Can of soda", 150),
    ]
    ctx_html = ""
    for icon, label, val in benchmarks:
        ok   = prediction >= val
        diff = abs(prediction - val)
        bg   = "#eaf3de" if ok else "#fbeaf0"
        fg   = "#3b6d11" if ok else "#993556"
        txt  = "covered ✓" if ok else f"{diff:.0f} kcal short"
        ctx_html += f"""
        <div class="ctx-row">
            <span>{icon} {label}
                <span style="color:#993556;font-family:'DM Mono',monospace;font-size:0.75rem;margin-left:6px;">{val} kcal</span>
            </span>
            <span style="background:{bg};color:{fg};padding:2px 10px;border-radius:10px;font-size:0.75rem;">{txt}</span>
        </div>"""
    st.markdown(ctx_html, unsafe_allow_html=True)

    # ── Performance table ──────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Performance Summary</div>', unsafe_allow_html=True)
    table = []
    for n in models.keys():
        table.append({
            "Algorithm":  ("✦ " if n == selected else "   ") + n,
            "Train MAE":  f"{perf[n]['train_mae']:.2f}",
            "Val MAE":    f"{perf[n]['val_mae']:.2f}",
            "Prediction": f"{all_preds[n]:.0f} kcal",
        })
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

    # ── Stats row ──────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(5)
    stats = [
        (f"{prediction:.0f}", "kcal Predicted"),
        (f"{bmi:.1f}",        f"BMI · {bmi_cat}"),
        (f"{duration} min",   "Duration"),
        (f"{heart_rate} bpm", "Heart Rate"),
        (f"{body_temp}°C",    "Body Temp"),
    ]
    for col, (val, lbl) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-top:2rem;color:#993556;font-size:0.7rem;
                font-family:'DM Mono',monospace;letter-spacing:1px;">
        🌸 trained on 15,000 exercise sessions · 5 algorithms compared
    </div>
    """, unsafe_allow_html=True)
