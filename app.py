import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CaloriFire · Burn Prediction Lab",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:       #0d1117;
    --surface:  #161b22;
    --surface2: #1e2530;
    --border:   #2a3140;
    --accent:   #ff6b35;
    --accent2:  #f7c59f;
    --teal:     #3ecfcf;
    --purple:   #a78bfa;
    --green:    #4ade80;
    --text:     #e8eaf0;
    --muted:    #8892a4;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #111827 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: 'Syne', sans-serif; }

/* Inputs */
.stSlider > div > div > div { background: var(--accent) !important; }
.stSelectbox > div > div { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #e84d0e) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2rem !important;
    letter-spacing: 0.5px;
    transition: all 0.2s;
    box-shadow: 0 4px 20px rgba(255,107,53,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(255,107,53,0.55) !important;
}

/* Metric cards */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: all 0.3s;
}
.metric-card:hover { border-color: var(--accent); transform: translateY(-3px); }
.metric-value { font-size: 2rem; font-weight: 800; color: var(--accent); }
.metric-label { font-size: 0.75rem; color: var(--muted); font-family: 'DM Mono', monospace; letter-spacing: 1px; text-transform: uppercase; margin-top: 4px; }

/* Hero title */
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 50%, var(--teal) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    letter-spacing: -1px;
}
.hero-sub { color: var(--muted); font-size: 1rem; margin-top: 0.5rem; font-family: 'DM Mono', monospace; }

/* Section header */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--teal);
    letter-spacing: 2px;
    text-transform: uppercase;
    border-left: 3px solid var(--teal);
    padding-left: 10px;
    margin-bottom: 1rem;
}

/* Algorithm card */
.algo-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.algo-name { font-weight: 700; font-size: 0.95rem; }
.algo-mae { font-family: 'DM Mono', monospace; color: var(--accent2); font-size: 0.85rem; }

/* Result banner */
.result-banner {
    background: linear-gradient(135deg, rgba(255,107,53,0.15), rgba(62,207,207,0.08));
    border: 1px solid rgba(255,107,53,0.4);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    text-align: center;
}
.result-cal { font-size: 4.5rem; font-weight: 800; color: var(--accent); line-height: 1; }
.result-unit { font-size: 1.1rem; color: var(--accent2); font-family: 'DM Mono', monospace; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# ─── Load & Train ─────────────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    exercise = pd.read_csv("exercise.csv")
    calories = pd.read_csv("calories.csv")
    df = exercise.merge(calories, on="User_ID")

    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])  # male=1, female=0

    features = df[["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]]
    target = df["Calories"].values

    X_train, X_val, Y_train, Y_val = train_test_split(
        features, target, test_size=0.1, random_state=22
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    model_defs = {
        "Linear Regression": LinearRegression(),
        "XGBoost":           XGBRegressor(random_state=42),
        "Lasso":             Lasso(),
        "Random Forest":     RandomForestRegressor(random_state=42),
        "Ridge":             Ridge(),
    }

    results = {}
    fitted  = {}
    for name, mdl in model_defs.items():
        mdl.fit(X_train_s, Y_train)
        train_mae = mean_absolute_error(Y_train, mdl.predict(X_train_s))
        val_mae   = mean_absolute_error(Y_val,   mdl.predict(X_val_s))
        results[name] = {"train_mae": round(train_mae, 2), "val_mae": round(val_mae, 2)}
        fitted[name]  = mdl

    return fitted, results, scaler, df


models, results, scaler, df = load_and_train()

# ─── Sidebar: User Input ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <span style="font-size:2.5rem">🔥</span>
        <div style="font-size:1.3rem; font-weight:800; color:#ff6b35; letter-spacing:-0.5px;">CaloriFire</div>
        <div style="font-size:0.7rem; color:#8892a4; font-family:'DM Mono',monospace; letter-spacing:1.5px;">BURN PREDICTION LAB</div>
    </div>
    <hr style="margin: 0.8rem 0;">
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Your Profile</div>', unsafe_allow_html=True)

    gender   = st.selectbox("Gender", ["Male", "Female"])
    age      = st.slider("Age (years)", 15, 80, 30)
    height   = st.slider("Height (cm)", 140, 210, 170)
    weight   = st.slider("Weight (kg)", 40, 150, 70)

    st.markdown('<div class="section-header" style="margin-top:1rem;">Exercise Stats</div>', unsafe_allow_html=True)

    duration   = st.slider("Duration (min)", 1, 120, 30)
    heart_rate = st.slider("Heart Rate (bpm)", 60, 200, 100)
    body_temp  = st.slider("Body Temp (°C)", 36.0, 42.0, 39.0, step=0.1)

    st.markdown('<div class="section-header" style="margin-top:1rem;">Algorithm</div>', unsafe_allow_html=True)
    selected_model = st.selectbox("Choose Algorithm", list(models.keys()))

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔥  Predict Calories", use_container_width=True)

# ─── Main Layout ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:0.3rem;">
    <div class="hero-title">Calories Burnt<br>Prediction Lab</div>
    <div class="hero-sub">// ML-powered exercise calorie estimator · 5 algorithms compared</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── Prediction ───────────────────────────────────────────────────────────────
gender_enc = 1 if gender == "Male" else 0
user_input = np.array([[gender_enc, age, height, weight, duration, heart_rate, body_temp]])
user_scaled = scaler.transform(user_input)

# Always show prediction (updates live)
prediction = models[selected_model].predict(user_scaled)[0]
prediction = max(0, prediction)

col_res, col_all = st.columns([1, 1.6], gap="large")

with col_res:
    st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-banner">
        <div style="color:#8892a4; font-size:0.75rem; font-family:'DM Mono',monospace; letter-spacing:2px; text-transform:uppercase; margin-bottom:0.4rem;">Estimated Burn</div>
        <div class="result-cal">{prediction:.0f}</div>
        <div class="result-unit">kilocalories</div>
        <div style="margin-top:1rem; color:#8892a4; font-size:0.75rem; font-family:'DM Mono',monospace;">via {selected_model}</div>
    </div>
    """, unsafe_allow_html=True)

    # Contextual comparison
    st.markdown("<br>", unsafe_allow_html=True)
    benchmarks = [("🍎 Apple", 95), ("🍕 Pizza slice", 285), ("🍔 Burger", 550), ("🏃 30-min run avg", 300)]
    st.markdown('<div class="section-header">Caloric Context</div>', unsafe_allow_html=True)
    for label, val in benchmarks:
        pct = min(prediction / val, 2.0)
        color = "#4ade80" if prediction >= val else "#f97316"
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem; background:#161b22; border-radius:8px; padding:0.5rem 0.8rem; border:1px solid #2a3140;">
            <span style="font-size:0.85rem;">{label}</span>
            <span style="font-family:'DM Mono',monospace; font-size:0.8rem; color:{color};">{val} kcal {"✓ covered" if prediction>=val else f"({val-prediction:.0f} short)"}</span>
        </div>""", unsafe_allow_html=True)

with col_all:
    st.markdown('<div class="section-header">Algorithm Comparison</div>', unsafe_allow_html=True)

    # Compute all predictions for this input
    all_preds = {}
    for name, mdl in models.items():
        p = max(0, mdl.predict(user_scaled)[0])
        all_preds[name] = p

    # MAE bar chart
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.patch.set_facecolor("#0d1117")

    names  = list(results.keys())
    t_maes = [results[n]["train_mae"] for n in names]
    v_maes = [results[n]["val_mae"]   for n in names]
    colors_t = ["#ff6b35" if n == selected_model else "#2a3140" for n in names]
    colors_v = ["#3ecfcf" if n == selected_model else "#1e3040" for n in names]

    x = np.arange(len(names))
    w = 0.38
    ax1 = axes[0]
    ax1.set_facecolor("#0d1117")
    ax1.barh(x + w/2, t_maes, w, color=colors_t, label="Train MAE")
    ax1.barh(x - w/2, v_maes, w, color=colors_v, label="Val MAE")
    ax1.set_yticks(x)
    ax1.set_yticklabels([n.replace(" ", "\n") for n in names], color="#e8eaf0", fontsize=8, fontfamily="monospace")
    ax1.set_xlabel("MAE (kcal)", color="#8892a4", fontsize=8)
    ax1.set_title("Model Error (MAE)", color="#e8eaf0", fontsize=10, fontweight="bold")
    ax1.tick_params(colors="#8892a4", labelsize=7)
    for spine in ax1.spines.values(): spine.set_color("#2a3140")
    ax1.xaxis.label.set_color("#8892a4")
    patch1 = mpatches.Patch(color="#ff6b35", label="Train MAE")
    patch2 = mpatches.Patch(color="#3ecfcf", label="Val MAE")
    ax1.legend(handles=[patch1, patch2], facecolor="#161b22", edgecolor="#2a3140", labelcolor="#e8eaf0", fontsize=7)
    ax1.grid(axis="x", color="#2a3140", linewidth=0.5)

    # Predictions bar
    ax2 = axes[1]
    ax2.set_facecolor("#0d1117")
    pred_vals  = [all_preds[n] for n in names]
    pred_colors = ["#ff6b35" if n == selected_model else "#2a3140" for n in names]
    bars = ax2.barh(names, pred_vals, color=pred_colors, edgecolor="#2a3140", linewidth=0.5)
    for bar, val in zip(bars, pred_vals):
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2,
                 f"{val:.0f}", va="center", ha="left", color="#e8eaf0", fontsize=7, fontfamily="monospace")
    ax2.set_yticklabels([n.replace(" ", "\n") for n in names], color="#e8eaf0", fontsize=8, fontfamily="monospace")
    ax2.set_xlabel("Predicted kcal", color="#8892a4", fontsize=8)
    ax2.set_title("Predictions — Your Input", color="#e8eaf0", fontsize=10, fontweight="bold")
    ax2.tick_params(colors="#8892a4", labelsize=7)
    for spine in ax2.spines.values(): spine.set_color("#2a3140")
    ax2.grid(axis="x", color="#2a3140", linewidth=0.5)

    plt.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Algorithm MAE table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Performance Summary</div>', unsafe_allow_html=True)
    
    rank_data = []
    for name in names:
        is_sel = "✦ " if name == selected_model else ""
        rank_data.append({
            "Algorithm": is_sel + name,
            "Train MAE": f"{results[name]['train_mae']:.2f}",
            "Val MAE": f"{results[name]['val_mae']:.2f}",
            "Prediction": f"{all_preds[name]:.1f} kcal"
        })
    rank_df = pd.DataFrame(rank_data)
    
    # Style it
    best_val = min(results[n]["val_mae"] for n in names)
    st.dataframe(
        rank_df,
        use_container_width=True,
        hide_index=True,
    )


# ─── Bottom: Stats ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">About Your Input</div>', unsafe_allow_html=True)

bmi = weight / ((height / 100) ** 2)
bmi_cat = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"

c1, c2, c3, c4, c5 = st.columns(5)
stats = [
    (c1, f"{bmi:.1f}", f"BMI · {bmi_cat}"),
    (c2, f"{prediction:.0f}", "kcal Predicted"),
    (c3, f"{duration} min", "Workout Duration"),
    (c4, f"{heart_rate} bpm", "Heart Rate"),
    (c5, f"{body_temp}°C", "Body Temp"),
]
for col, val, label in stats:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#8892a4; font-size:0.7rem; font-family:'DM Mono',monospace; letter-spacing:1px;">
    CALORIFIRE · TRAINED ON 15,000 EXERCISE SESSIONS · 5 ALGORITHMS COMPARED
</div>""", unsafe_allow_html=True)
