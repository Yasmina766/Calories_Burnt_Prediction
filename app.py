import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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

# ── Pink CSS theme ─────────────────────────────────────────────────────────────
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

.stSlider > div > div > div { background: #d4537e !important; }
.stSlider > div > div > div > div { background: #d4537e !important; }

.stSelectbox > div > div {
    background: white !important;
    border: 1px solid #f4c0d1 !important;
    border-radius: 10px !important;
}

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
}
.stButton > button:hover { transform: translateY(-2px); }

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

.ctx-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #fbeaf0;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ── Load & Train ───────────────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    exercise = pd.read_csv("exercise.csv")
    calories = pd.read_csv("calories.csv")
    df = exercise.merge(calories, on="User_ID")

    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])   # female=0, male=1

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
  <div style="font-size:2.6rem; font-weight:600; color:#4b1528; letter-spacing:-1px;">
    🌸 Calories Burnt Predictor
  </div>
  <div style="font-size:0.85rem; color:#993556; font-family:'DM Mono',monospace; margin-top:6px;">
    Enter your data · choose an algorithm · see the result
  </div>
</div>
""", unsafe_allow_html=True)

# ── Inputs ─────────────────────────────────────────────────────────────────────
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

# ── Algorithm selector ─────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">Choose Algorithm</div>', unsafe_allow_html=True)
selected = st.radio("", list(models.keys()), horizontal=True, label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)
st.button("🌸  Predict Calories Burned")   # triggers re-run automatically

# ── Compute predictions ────────────────────────────────────────────────────────
gender_enc  = 1 if gender == "Male" else 0
user_input  = np.array([[gender_enc, age, height, weight, duration, heart_rate, body_temp]])
user_scaled = scaler.transform(user_input)

prediction  = max(0.0, float(models[selected].predict(user_scaled)[0]))
all_preds   = {n: max(0.0, float(m.predict(user_scaled)[0])) for n, m in models.items()}

# ── Result card ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="sec-label">Prediction Result</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="result-wrap">
    <div style="font-size:0.72rem;color:#993556;font-family:'DM Mono',monospace;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">
        Estimated Burn
    </div>
    <div class="result-num">{prediction:.0f}</div>
    <div class="result-unit">kilocalories</div>
    <div class="result-algo">via {selected}</div>
</div>
""", unsafe_allow_html=True)

# BMI
bmi     = weight / ((height / 100) ** 2)
bmi_cat = "Underweight" if bmi < 18.5 else "Normal weight" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
bmi_col = "#3b6d11" if bmi < 25 else "#ba7517" if bmi < 30 else "#a32d2d"
bmi_bg  = "#eaf3de" if bmi < 25 else "#faeeda" if bmi < 30 else "#fcebeb"
st.markdown(f"""
<div style="text-align:center;margin-top:10px;font-size:0.85rem;color:#72243e;">
    BMI: <strong>{bmi:.1f}</strong>
    <span style="background:{bmi_bg};color:{bmi_col};padding:2px 10px;border-radius:20px;font-size:0.75rem;margin-left:6px;">
        {bmi_cat}
    </span>
</div>
""", unsafe_allow_html=True)

# Caloric context
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
    badge_style = "background:#eaf3de;color:#3b6d11;" if ok else "background:#fbeaf0;color:#993556;"
    badge_text  = "covered ✓" if ok else f"{diff:.0f} kcal short"
    ctx_html += f"""
    <div class="ctx-row">
        <span>{icon} {label}
            <span style="color:#993556;font-family:'DM Mono',monospace;font-size:0.75rem;margin-left:6px;">{val} kcal</span>
        </span>
        <span style="{badge_style}padding:2px 10px;border-radius:10px;font-size:0.75rem;">{badge_text}</span>
    </div>"""
st.markdown(ctx_html, unsafe_allow_html=True)

# ── Charts ─────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="sec-label">Algorithm Comparison</div>', unsafe_allow_html=True)

names  = list(perf.keys())
v_maes = [perf[n]["val_mae"]   for n in names]
t_maes = [perf[n]["train_mae"] for n in names]
preds  = [all_preds[n]         for n in names]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
fig.patch.set_facecolor("white")

c_active  = "#d4537e"
c_passive = "#f4c0d1"
c_teal_a  = "#3ecfcf"
c_teal_p  = "#b2f0f0"

x = np.arange(len(names))
w = 0.35

# MAE chart
ax1.set_facecolor("white")
ax1.barh(x + w/2, t_maes, w, color=[c_active if n==selected else c_passive for n in names])
ax1.barh(x - w/2, v_maes, w, color=[c_teal_a if n==selected else c_teal_p  for n in names])
ax1.set_yticks(x)
ax1.set_yticklabels([n.replace(" ", "\n") for n in names], fontsize=8, color="#4b1528")
ax1.set_xlabel("MAE (kcal)", fontsize=8, color="#993556")
ax1.set_title("Model Error (MAE)", fontsize=10, fontweight="bold", color="#4b1528", pad=10)
ax1.tick_params(colors="#993556", labelsize=7)
for sp in ax1.spines.values(): sp.set_color("#f4c0d1")
ax1.grid(axis="x", color="#fbeaf0", linewidth=0.8)
ax1.legend(
    handles=[Patch(color=c_active, label="Train MAE"), Patch(color=c_teal_a, label="Val MAE")],
    facecolor="white", edgecolor="#f4c0d1", labelcolor="#4b1528", fontsize=7
)

# Predictions chart
ax2.set_facecolor("white")
bars = ax2.barh(names, preds,
                color=[c_active if n==selected else c_passive for n in names],
                edgecolor="#f4c0d1", linewidth=0.5)
for bar, val in zip(bars, preds):
    ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2,
             f"{val:.0f}", va="center", ha="left", fontsize=7,
             color="#4b1528", fontfamily="monospace")
ax2.set_yticklabels([n.replace(" ", "\n") for n in names], fontsize=8, color="#4b1528")
ax2.set_xlabel("Predicted kcal", fontsize=8, color="#993556")
ax2.set_title("Predictions — Your Input", fontsize=10, fontweight="bold", color="#4b1528", pad=10)
ax2.tick_params(colors="#993556", labelsize=7)
for sp in ax2.spines.values(): sp.set_color("#f4c0d1")
ax2.grid(axis="x", color="#fbeaf0", linewidth=0.8)

plt.tight_layout(pad=1.8)
st.pyplot(fig, use_container_width=True)
plt.close()

# ── Performance table ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="sec-label">Performance Summary</div>', unsafe_allow_html=True)
table = []
for n in names:
    table.append({
        "Algorithm":  ("✦ " if n == selected else "   ") + n,
        "Train MAE":  f"{perf[n]['train_mae']:.2f}",
        "Val MAE":    f"{perf[n]['val_mae']:.2f}",
        "Prediction": f"{all_preds[n]:.0f} kcal",
    })
st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

# ── Stats row ──────────────────────────────────────────────────────────────────
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
