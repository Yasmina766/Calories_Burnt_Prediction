import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Calories Burnt Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #FF6B35, #F7C59F);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #888;
    font-size: 1rem;
    margin-bottom: 2rem;
}

.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #FF6B35;
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
    color: white;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #FF6B35;
}

.metric-label {
    font-size: 0.8rem;
    color: #aaa;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.result-box {
    background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    color: white;
    box-shadow: 0 8px 32px rgba(255, 107, 53, 0.3);
}

.result-calories {
    font-size: 4rem;
    font-weight: 700;
    line-height: 1;
}

.result-unit {
    font-size: 1.2rem;
    opacity: 0.85;
}

.algo-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    margin-top: 0.5rem;
}

.stSelectbox label, .stSlider label {
    font-weight: 500;
    color: #333;
}

.sidebar-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #FF6B35;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #FF6B35;
}

div[data-testid="stMetric"] {
    background: #fff8f5;
    border: 1px solid #ffe0d0;
    border-radius: 12px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_train():
    exercise = pd.read_csv("exercise.csv")
    calories = pd.read_csv("calories__2_.csv")

    df = exercise.merge(calories, on="User_ID")
    df.dropna(inplace=True)

    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])

    features = df[["Gender", "Age", "Height", "Heart_Rate", "Body_Temp", "Weight", "Duration"]]
    target = df["Calories"].values

    X_train, X_val, Y_train, Y_val = train_test_split(
        features, target, test_size=0.1, random_state=22
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    models = {
        "XGBoost": XGBRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, Y_train)
        val_preds = model.predict(X_val_scaled)
        mae = mean_absolute_error(Y_val, val_preds)
        results[name] = {"model": model, "mae": mae}

    return results, scaler, df


with st.spinner("Training models on your dataset..."):
    model_results, scaler, df = load_and_train()

st.markdown('<div class="main-title">🔥 Calories Burnt Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning — 5 algorithms trained on your exercise dataset</div>', unsafe_allow_html=True)

col_info1, col_info2, col_info3, col_info4 = st.columns(4)
with col_info1:
    st.metric("Dataset Size", f"{len(df):,} rows")
with col_info2:
    st.metric("Features Used", "7 features")
with col_info3:
    best_model = min(model_results, key=lambda k: model_results[k]["mae"])
    st.metric("Best Model", best_model)
with col_info4:
    best_mae = model_results[best_model]["mae"]
    st.metric("Best MAE", f"±{best_mae:.1f} kcal")

st.divider()

with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Your Details</div>', unsafe_allow_html=True)

    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", min_value=20, max_value=79, value=30, step=1)
    height = st.slider("Height (cm)", min_value=123, max_value=222, value=170, step=1)
    weight = st.slider("Weight (kg)", min_value=36, max_value=132, value=70, step=1)

    st.markdown("---")
    st.markdown('<div class="sidebar-header">🏋️ Workout Info</div>', unsafe_allow_html=True)

    duration = st.slider("Duration (min)", min_value=1, max_value=30, value=15, step=1)
    heart_rate = st.slider("Heart Rate (bpm)", min_value=67, max_value=128, value=95, step=1)
    body_temp = st.slider("Body Temperature (°C)", min_value=37.1, max_value=41.5, value=39.0, step=0.1)

    st.markdown("---")
    st.markdown('<div class="sidebar-header">🤖 Algorithm</div>', unsafe_allow_html=True)

    algo_names = list(model_results.keys())
    algo_display = [f"{n}  (MAE ±{model_results[n]['mae']:.1f})" for n in algo_names]
    selected_display = st.selectbox("Choose Algorithm", algo_display)
    selected_algo = algo_names[algo_display.index(selected_display)]

    predict_btn = st.button("🔥 Predict Calories", use_container_width=True, type="primary")

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("📊 Model Performance")
    mae_df = pd.DataFrame({
        "Algorithm": algo_names,
        "MAE (kcal)": [model_results[n]["mae"] for n in algo_names]
    }).sort_values("MAE (kcal)")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = ["#FF6B35" if n == best_model else "#f0c4b0" for n in mae_df["Algorithm"]]
    bars = ax.barh(mae_df["Algorithm"], mae_df["MAE (kcal)"], color=colors, height=0.5)
    ax.set_xlabel("Mean Absolute Error (kcal)", fontsize=10)
    ax.set_title("Validation MAE by Algorithm", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, mae_df["MAE (kcal)"]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f"±{val:.1f}", va="center", fontsize=9, color="#555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

with right_col:
    st.subheader("📈 Data Insights")
    fig2, axes = plt.subplots(1, 2, figsize=(6, 3.5))

    axes[0].hist(df["Calories"], bins=30, color="#FF6B35", alpha=0.8, edgecolor="white")
    axes[0].set_title("Calories Distribution", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Calories (kcal)")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    corr_cols = ["Age", "Height", "Heart_Rate", "Body_Temp", "Weight", "Duration", "Calories"]
    corr = df[corr_cols].corr()["Calories"].drop("Calories").sort_values()
    colors2 = ["#FF6B35" if v > 0 else "#aaa" for v in corr.values]
    axes[1].barh(corr.index, corr.values, color=colors2, height=0.5)
    axes[1].set_title("Feature Correlation", fontsize=10, fontweight="bold")
    axes[1].axvline(0, color="#ccc", linewidth=0.8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close()

st.divider()

if predict_btn:
    gender_encoded = 1 if gender == "Male" else 0
    input_data = np.array([[gender_encoded, age, height, heart_rate, body_temp, weight, duration]])
    input_scaled = scaler.transform(input_data)

    model = model_results[selected_algo]["model"]
    prediction = model.predict(input_scaled)[0]
    prediction = max(1, round(prediction))
    mae = model_results[selected_algo]["mae"]

    all_preds = {n: max(1, round(model_results[n]["model"].predict(input_scaled)[0]))
                 for n in algo_names}

    bmi = weight / ((height / 100) ** 2)
    rate_per_min = prediction / duration

    intensity = "Low 🟢" if prediction < 80 else ("High 🔴" if prediction > 180 else "Moderate 🟡")

    res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1])

    with res_col1:
        st.markdown(f"""
        <div class="result-box">
            <div style="font-size:0.9rem;opacity:0.8;margin-bottom:0.5rem">PREDICTED CALORIES</div>
            <div class="result-calories">{prediction}</div>
            <div class="result-unit">kcal burned</div>
            <div class="algo-badge">{selected_algo}</div>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        st.metric("BMI", f"{bmi:.1f}")
        st.metric("kcal / min", f"{rate_per_min:.1f}")

    with res_col3:
        st.metric("Intensity", intensity)
        st.metric("Model MAE", f"±{mae:.1f} kcal")

    st.markdown("#### 🤖 All Models Comparison")
    comp_df = pd.DataFrame({
        "Algorithm": list(all_preds.keys()),
        "Predicted Calories (kcal)": list(all_preds.values()),
        "MAE": [f"±{model_results[n]['mae']:.1f}" for n in all_preds.keys()]
    }).sort_values("Predicted Calories (kcal)", ascending=False)
    comp_df["Selected"] = comp_df["Algorithm"] == selected_algo

    fig3, ax3 = plt.subplots(figsize=(8, 3))
    bar_colors = ["#FF6B35" if n == selected_algo else "#f0c4b0" for n in comp_df["Algorithm"]]
    ax3.barh(comp_df["Algorithm"], comp_df["Predicted Calories (kcal)"], color=bar_colors, height=0.5)
    for i, (val, name) in enumerate(zip(comp_df["Predicted Calories (kcal)"], comp_df["Algorithm"])):
        ax3.text(val + 1, i, f"{val} kcal", va="center", fontsize=9)
    ax3.set_xlabel("Predicted Calories (kcal)")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close()
