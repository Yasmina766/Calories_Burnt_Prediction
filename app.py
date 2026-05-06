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
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Calories Burnt Predictor",
    page_icon="🔥",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

/* Header */
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ff6b35, #f7931e, #ff6b35);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    margin-bottom: 0;
}

.hero-sub {
    color: #888;
    font-size: 1rem;
    margin-top: 4px;
    font-weight: 300;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Cards */
.metric-card {
    background: #13131a;
    border: 1px solid #222233;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #ff6b35; }
.metric-card .label {
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #ff6b35;
}
.metric-card .sub {
    font-size: 0.75rem;
    color: #555;
    margin-top: 2px;
}

/* Algorithm pill */
.algo-badge {
    display: inline-block;
    background: #1a1a28;
    border: 1px solid #ff6b35;
    color: #ff6b35;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 16px;
}

/* Result big number */
.result-box {
    background: linear-gradient(135deg, #1a0d05, #1f1200);
    border: 2px solid #ff6b35;
    border-radius: 20px;
    padding: 32px;
    text-align: center;
}
.result-box .kcal-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 4rem;
    font-weight: 700;
    color: #ff6b35;
    line-height: 1;
}
.result-box .kcal-unit {
    font-size: 1.2rem;
    color: #888;
    margin-top: 6px;
}
.result-box .result-label {
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 12px;
}

/* Comparison table */
.cmp-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 16px;
    border-radius: 10px;
    margin-bottom: 8px;
    background: #13131a;
    border: 1px solid #1e1e2e;
    font-size: 0.9rem;
}
.cmp-row.best {
    border-color: #ff6b35;
    background: #1a1005;
}
.cmp-bar-wrap {
    flex: 1;
    margin: 0 16px;
    background: #1e1e2e;
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}
.cmp-bar {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #ff6b35, #f7931e);
}
.cmp-mae {
    font-family: 'JetBrains Mono', monospace;
    color: #aaa;
    font-size: 0.8rem;
    width: 70px;
    text-align: right;
}
.cmp-algo {
    width: 160px;
    font-size: 0.8rem;
}

/* Sidebar inputs */
section[data-testid="stSidebar"] {
    background: #0d0d14;
    border-right: 1px solid #1e1e2e;
}
section[data-testid="stSidebar"] .stSlider > div > div {
    color: #ff6b35;
}

/* Streamlit overrides */
.stSelectbox > div > div {
    background: #13131a;
    border: 1px solid #2a2a3e;
    border-radius: 10px;
    color: #e8e8f0;
}
.stButton > button {
    background: linear-gradient(135deg, #ff6b35, #f7931e);
    color: #000;
    font-weight: 700;
    border: none;
    border-radius: 12px;
    padding: 12px 32px;
    font-size: 1rem;
    letter-spacing: 0.5px;
    font-family: 'Space Grotesk', sans-serif;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
hr { border-color: #1e1e2e; }

div[data-testid="stMetric"] {
    background: #13131a;
    border: 1px solid #222233;
    border-radius: 12px;
    padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading & model training ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️  Training models on your data…")
def load_and_train():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    df = exercise.merge(calories, on="User_ID")

    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])   # male=1, female=0

    to_remove = ["Weight", "Duration"]
    df.drop(to_remove, axis=1, inplace=True)

    features = df.drop(["User_ID", "Calories"], axis=1)
    target   = df["Calories"].values

    X_train, X_val, y_train, y_val = train_test_split(
        features, target, test_size=0.1, random_state=22
    )

    scaler  = StandardScaler()
    Xtr     = scaler.fit_transform(X_train)
    Xvl     = scaler.transform(X_val)

    algos = {
        "Linear Regression":    LinearRegression(),
        "Lasso":                Lasso(),
        "Ridge":                Ridge(),
        "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost":              XGBRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    for name, mdl in algos.items():
        mdl.fit(Xtr, y_train)
        train_mae = mae(y_train, mdl.predict(Xtr))
        val_mae   = mae(y_val,   mdl.predict(Xvl))
        results[name] = {
            "model":     mdl,
            "train_mae": train_mae,
            "val_mae":   val_mae,
        }

    feature_names = list(features.columns)
    return results, scaler, feature_names, df


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


# ── Load ───────────────────────────────────────────────────────────────────────
try:
    results, scaler, feature_names, df = load_and_train()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False


# ── Header ─────────────────────────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown('<p class="hero-title">🔥 CalorieSense</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Calories Burnt Prediction · ML Algorithm Benchmark</p>', unsafe_allow_html=True)

st.markdown("---")

if not data_loaded:
    st.error("❌  Could not find **calories.csv** and **exercise.csv** — make sure they're in the same folder as `app.py`.")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Your Profile")
    st.markdown("---")

    gender   = st.selectbox("Gender", ["Male", "Female"])
    age      = st.slider("Age",        10, 80, 25)
    height   = st.slider("Height (cm)", 140, 210, 170)
    heart_rate = st.slider("Heart Rate (bpm)", 60, 130, 90)
    body_temp  = st.slider("Body Temp (°C)",   37.0, 42.0, 39.5, step=0.1)

    st.markdown("---")
    st.markdown("### 🤖 Algorithm")
    selected_algo = st.selectbox(
        "Choose model",
        list(results.keys()),
        index=4,   # XGBoost default
    )
    st.markdown("---")
    predict_btn = st.button("🔥 Predict Calories")

# ── Main layout ────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

# ── LEFT: Prediction ───────────────────────────────────────────────────────────
with left:
    st.markdown("#### Prediction")

    gender_enc = 1 if gender == "Male" else 0
    # feature order: Gender, Age, Height, Heart_Rate, Body_Temp
    user_input = np.array([[gender_enc, age, height, heart_rate, body_temp]])
    user_scaled = scaler.transform(user_input)

    model = results[selected_algo]["model"]
    prediction = max(0, model.predict(user_scaled)[0])

    st.markdown(f'<div class="algo-badge">{selected_algo}</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Estimated Calories Burnt</div>
        <div class="kcal-value">{prediction:.0f}</div>
        <div class="kcal-unit">kcal</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Input summary cards
    pairs = [
        ("Gender",      gender,                    ""),
        ("Age",         f"{age}",                  "years"),
        ("Height",      f"{height}",               "cm"),
        ("Heart Rate",  f"{heart_rate}",            "bpm"),
        ("Body Temp",   f"{body_temp:.1f}",         "°C"),
    ]
    c1, c2 = st.columns(2)
    for i, (label, val, unit) in enumerate(pairs):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{val}</div>
                <div class="sub">{unit}</div>
            </div>
            """, unsafe_allow_html=True)

# ── RIGHT: Algorithm comparison ────────────────────────────────────────────────
with right:
    st.markdown("#### Algorithm Benchmark")
    st.markdown('<span style="color:#555;font-size:0.8rem;">Validation MAE — lower is better</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    max_mae = max(v["val_mae"] for v in results.values())
    sorted_algos = sorted(results.items(), key=lambda x: x[1]["val_mae"])

    for name, info in sorted_algos:
        val_mae   = info["val_mae"]
        train_mae = info["train_mae"]
        bar_pct   = (1 - val_mae / max_mae) * 100
        is_best   = name == sorted_algos[0][0]
        is_sel    = name == selected_algo
        badge     = " 🏆" if is_best else (" ◀" if is_sel else "")
        row_cls   = "cmp-row best" if is_best else "cmp-row"

        st.markdown(f"""
        <div class="{row_cls}">
            <div class="cmp-algo">{name}{badge}</div>
            <div class="cmp-bar-wrap">
                <div class="cmp-bar" style="width:{bar_pct:.1f}%"></div>
            </div>
            <div class="cmp-mae">{val_mae:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed stats for selected model
    sel = results[selected_algo]
    st.markdown(f"**{selected_algo} — Detailed Stats**")
    m1, m2 = st.columns(2)
    m1.metric("Train MAE", f"{sel['train_mae']:.2f} kcal")
    m2.metric("Val MAE",   f"{sel['val_mae']:.2f} kcal")

    overfit_gap = sel["train_mae"] - sel["val_mae"]
    if abs(overfit_gap) < 2:
        fit_note = "✅ Well-generalised"
    elif overfit_gap < 0:
        fit_note = "⚠️ Slight underfitting"
    else:
        fit_note = "⚠️ Signs of overfitting"
    st.caption(fit_note)

st.markdown("---")

# ── Dataset overview ──────────────────────────────────────────────────────────
with st.expander("📊 Dataset Overview"):
    st.markdown(f"**{len(df):,} records** · 5 features after preprocessing")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Calories",    f"{df['Calories'].mean():.0f} kcal")
    c2.metric("Min Calories",    f"{df['Calories'].min():.0f} kcal")
    c3.metric("Max Calories",    f"{df['Calories'].max():.0f} kcal")
    c4.metric("Std Dev",         f"{df['Calories'].std():.0f} kcal")
    st.dataframe(df.drop("User_ID", axis=1).describe().round(2), use_container_width=True)

st.markdown(
    '<p style="text-align:center;color:#333;font-size:0.75rem;margin-top:24px;">'
    'CalorieSense · Built with Streamlit & scikit-learn</p>',
    unsafe_allow_html=True
)
