import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Calories Burnt Prediction",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3a);
        border: 1px solid #2e3450;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-card h3 { color: #a0aec0; font-size: 0.85rem; margin: 0; }
    .metric-card h1 { color: #f6ad55; font-size: 2rem; margin: 4px 0 0; }
    .algo-card {
        background: linear-gradient(135deg, #1a1f2e, #1e2436);
        border: 1px solid #2d3556;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 10px;
    }
    .algo-title { color: #63b3ed; font-weight: bold; font-size: 1rem; }
    .algo-desc  { color: #718096; font-size: 0.82rem; margin-top: 4px; }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f6ad55;
        border-left: 4px solid #f6ad55;
        padding-left: 12px;
        margin: 24px 0 16px;
    }
    .stTabs [data-baseweb="tab"] { color: #a0aec0; }
    .stTabs [aria-selected="true"] { color: #f6ad55 !important; border-bottom-color: #f6ad55 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Load & Prepare Data ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    cal = pd.read_csv("calories.csv")
    ex  = pd.read_csv("exercise.csv")
    df  = ex.merge(cal, on="User_ID")
    df.replace({"male": 0, "female": 1}, inplace=True)
    # Drop highly correlated features (same as notebook: Weight & Duration)
    df.drop(columns=["Weight", "Duration"], inplace=True)
    return df

@st.cache_data
def train_models(df):
    features = df.drop(["User_ID", "Calories"], axis=1)
    target   = df["Calories"].values

    X_train, X_val, y_train, y_val = train_test_split(
        features, target, test_size=0.1, random_state=22
    )

    scaler  = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_val   = scaler.transform(X_val)

    model_defs = {
        "Linear Regression": LinearRegression(),
        "Lasso":             Lasso(),
        "Ridge":             Ridge(),
        "Random Forest":     RandomForestRegressor(random_state=42),
        "XGBoost":           XGBRegressor(verbosity=0, random_state=42),
    }

    results = {}
    for name, mdl in model_defs.items():
        mdl.fit(Xs_train, y_train)
        tr_pred  = mdl.predict(Xs_train)
        val_pred = mdl.predict(Xs_val)
        results[name] = {
            "model":       mdl,
            "scaler":      scaler,
            "train_mae":   mae_score(y_train, tr_pred),
            "val_mae":     mae_score(y_val,   val_pred),
            "train_r2":    r2_score(y_train,  tr_pred),
            "val_r2":      r2_score(y_val,    val_pred),
            "val_pred":    val_pred,
            "y_val":       y_val,
        }
    return results, features.columns.tolist()

def mae_score(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# ─── Descriptions for each algorithm ────────────────────────────────────────────
ALGO_INFO = {
    "Linear Regression": {
        "desc": "يجد أفضل خط مستقيم يصف العلاقة بين المتغيرات والسعرات المحروقة.",
        "pros": "سريع، سهل التفسير، يعمل بشكل ممتاز على البيانات الخطية.",
        "cons": "لا يلتقط العلاقات غير الخطية.",
        "color": "#4299e1",
    },
    "Lasso": {
        "desc": "Regularized Linear Regression تُضيف عقوبة L1 لتقليل التعقيد وانتقاء المتغيرات تلقائياً.",
        "pros": "يُلغي المتغيرات الضعيفة تلقائياً، يقلل الـ overfitting.",
        "cons": "قد يُفرط في حذف بعض المتغيرات المفيدة.",
        "color": "#9f7aea",
    },
    "Ridge": {
        "desc": "Regularized Linear Regression تُضيف عقوبة L2 لتوزيع الأوزان بشكل متوازن.",
        "pros": "يعالج الـ multicollinearity بشكل ممتاز.",
        "cons": "لا يُلغي المتغيرات، يحتفظ بها جميعاً بأوزان صغيرة.",
        "color": "#ed64a6",
    },
    "Random Forest": {
        "desc": "مجموعة من أشجار القرار تعمل معاً وتتوسط نتائجها لتحسين الدقة.",
        "pros": "دقيق جداً، يتعامل مع العلاقات غير الخطية، مقاوم لـ overfitting.",
        "cons": "بطيء في التدريب، صعب التفسير.",
        "color": "#48bb78",
    },
    "XGBoost": {
        "desc": "Gradient Boosting متطور يبني نماذج متتالية كل منها يصحح أخطاء السابق.",
        "pros": "من أقوى الخوارزميات، سريع، يتعامل مع القيم المفقودة.",
        "cons": "يحتاج ضبط دقيق لـ hyperparameters.",
        "color": "#f6ad55",
    },
}

# ─── Main App ───────────────────────────────────────────────────────────────────
def main():
    # ── Header
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <h1 style="font-size:2.6rem; color:#f6ad55; margin:0;">🔥 Calories Burnt Prediction</h1>
        <p style="color:#718096; font-size:1rem; margin-top:6px;">
            Machine Learning Dashboard — تحليل وتوقع السعرات الحرارية المحروقة أثناء التمرين
        </p>
    </div>
    <hr style="border-color:#2d3556; margin-bottom:24px;">
    """, unsafe_allow_html=True)

    # ── Load
    with st.spinner("جاري تحميل البيانات وتدريب النماذج…"):
        df      = load_data()
        results, feature_cols = train_models(df)

    # ── Sidebar — Prediction Panel
    st.sidebar.markdown("## 🎯 توقع السعرات المحروقة")
    st.sidebar.markdown("أدخل بيانات التمرين للحصول على توقع فوري.")

    gender     = st.sidebar.selectbox("الجنس", ["ذكر", "أنثى"])
    age        = st.sidebar.slider("العمر", 18, 79, 30)
    height     = st.sidebar.slider("الطول (سم)", 120, 220, 170)
    heart_rate = st.sidebar.slider("معدل ضربات القلب", 50, 180, 100)
    body_temp  = st.sidebar.slider("درجة حرارة الجسم (°C)", 36.0, 42.0, 38.5, step=0.1)

    sel_model = st.sidebar.selectbox("اختر النموذج", list(results.keys()))

    if st.sidebar.button("🔥 احسب السعرات", use_container_width=True):
        gender_val = 1 if gender == "أنثى" else 0
        inp = np.array([[gender_val, age, height, heart_rate, body_temp]])
        mdl_data = results[sel_model]
        inp_scaled = mdl_data["scaler"].transform(inp)
        pred = mdl_data["model"].predict(inp_scaled)[0]
        st.sidebar.markdown(f"""
        <div style="background:linear-gradient(135deg,#2d1b0e,#3d2510);
                    border:1px solid #f6ad55; border-radius:12px; padding:20px; text-align:center; margin-top:16px;">
            <p style="color:#f6ad55;font-size:0.9rem;margin:0;">السعرات المحروقة المتوقعة</p>
            <h1 style="color:#fff;font-size:2.8rem;margin:4px 0;">{pred:.1f}</h1>
            <p style="color:#a0aec0;font-size:0.8rem;margin:0;">سعرة حرارية 🔥</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 نظرة عامة على البيانات",
        "🤖 الخوارزميات",
        "📈 مقارنة الأداء",
        "🔍 تحليل التوقعات",
    ])

    # ══════════════════════════════════════════════════════════
    # TAB 1 — Data Overview
    # ══════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-header">إحصائيات البيانات</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        for col, label, val in zip(
            [col1, col2, col3, col4],
            ["عدد السجلات", "المتغيرات", "متوسط السعرات", "الحد الأقصى للسعرات"],
            [len(df), len(feature_cols), f"{df['Calories'].mean():.1f}", f"{df['Calories'].max():.0f}"],
        ):
            col.markdown(f"""
            <div class="metric-card">
                <h3>{label}</h3><h1>{val}</h1>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">توزيع البيانات</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            # Calories distribution
            fig = px.histogram(df, x="Calories", nbins=50, title="توزيع السعرات المحروقة",
                               color_discrete_sequence=["#f6ad55"])
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            # Gender pie
            original_ex = pd.read_csv("exercise.csv")
            gender_counts = original_ex["Gender"].value_counts()
            fig2 = px.pie(values=gender_counts.values,
                          names=["ذكر" if g == "male" else "أنثى" for g in gender_counts.index],
                          title="توزيع الجنس", color_discrete_sequence=["#63b3ed", "#f687b3"])
            fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">العلاقة بين المتغيرات والسعرات</div>', unsafe_allow_html=True)
        sample = df.sample(min(1000, len(df)), random_state=1)
        feat_options = ["Age", "Height", "Heart_Rate", "Body_Temp"]
        fig3 = make_subplots(rows=2, cols=2,
                             subplot_titles=["العمر", "الطول", "معدل القلب", "حرارة الجسم"])
        colors = ["#4299e1", "#9f7aea", "#48bb78", "#f6ad55"]
        for i, (feat, clr) in enumerate(zip(feat_options, colors)):
            r, c = divmod(i, 2)
            fig3.add_trace(
                go.Scatter(x=sample[feat], y=sample["Calories"], mode="markers",
                           marker=dict(color=clr, size=4, opacity=0.5),
                           name=feat),
                row=r+1, col=c+1,
            )
        fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", height=500, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown('<div class="section-header">عينة من البيانات</div>', unsafe_allow_html=True)
        display_df = df.copy()
        display_df["Gender"] = display_df["Gender"].map({0: "ذكر", 1: "أنثى"})
        st.dataframe(display_df.head(10), use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # TAB 2 — Algorithms
    # ══════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-header">الخوارزميات المستخدمة في المشروع</div>',
                    unsafe_allow_html=True)

        for name, info in ALGO_INFO.items():
            r = results[name]
            col_left, col_right = st.columns([2, 1])
            with col_left:
                st.markdown(f"""
                <div class="algo-card">
                    <div class="algo-title" style="color:{info['color']};">⚡ {name}</div>
                    <div class="algo-desc">{info['desc']}</div>
                    <div style="margin-top:10px; display:flex; gap:20px; flex-wrap:wrap;">
                        <span style="color:#48bb78; font-size:0.8rem;">✅ {info['pros']}</span>
                        <span style="color:#fc8181; font-size:0.8rem;">⚠️ {info['cons']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_right:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.03); border:1px solid #2d3556;
                            border-radius:10px; padding:14px; text-align:center; margin-top:4px;">
                    <p style="color:#a0aec0;font-size:0.75rem;margin:0;">Validation MAE</p>
                    <h2 style="color:{info['color']};margin:4px 0;">{r['val_mae']:.2f}</h2>
                    <p style="color:#a0aec0;font-size:0.75rem;margin:0;">R² Score</p>
                    <h2 style="color:{info['color']};margin:4px 0;">{r['val_r2']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # TAB 3 — Performance Comparison
    # ══════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-header">مقارنة أداء الخوارزميات</div>', unsafe_allow_html=True)

        names      = list(results.keys())
        train_maes = [results[n]["train_mae"] for n in names]
        val_maes   = [results[n]["val_mae"]   for n in names]
        val_r2s    = [results[n]["val_r2"]    for n in names]
        colors_list = [ALGO_INFO[n]["color"]  for n in names]

        col1, col2 = st.columns(2)
        with col1:
            fig_mae = go.Figure()
            fig_mae.add_trace(go.Bar(name="Training MAE", x=names, y=train_maes,
                                     marker_color="#63b3ed"))
            fig_mae.add_trace(go.Bar(name="Validation MAE", x=names, y=val_maes,
                                     marker_color="#f6ad55"))
            fig_mae.update_layout(title="Mean Absolute Error (MAE) — Training vs Validation",
                                  barmode="group", template="plotly_dark",
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  yaxis_title="MAE (Calories)", xaxis_tickangle=-15)
            st.plotly_chart(fig_mae, use_container_width=True)

        with col2:
            fig_r2 = go.Figure(go.Bar(x=names, y=val_r2s, marker_color=colors_list,
                                      text=[f"{v:.4f}" for v in val_r2s],
                                      textposition="outside"))
            fig_r2.update_layout(title="R² Score (Validation) — كلما اقترب من 1 كلما كان أفضل",
                                 template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(0,0,0,0)",
                                 yaxis=dict(title="R² Score", range=[0, 1.1]),
                                 xaxis_tickangle=-15)
            st.plotly_chart(fig_r2, use_container_width=True)

        # Summary table
        st.markdown('<div class="section-header">جدول ملخص النتائج</div>', unsafe_allow_html=True)
        summary = pd.DataFrame({
            "الخوارزمية":      names,
            "Train MAE":      [f"{results[n]['train_mae']:.3f}" for n in names],
            "Validation MAE": [f"{results[n]['val_mae']:.3f}"   for n in names],
            "Train R²":       [f"{results[n]['train_r2']:.4f}"  for n in names],
            "Validation R²":  [f"{results[n]['val_r2']:.4f}"   for n in names],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        # Best model callout
        best = min(results, key=lambda n: results[n]["val_mae"])
        st.success(f"🏆 أفضل نموذج بناءً على Validation MAE: **{best}** "
                   f"(MAE = {results[best]['val_mae']:.3f})")

    # ══════════════════════════════════════════════════════════
    # TAB 4 — Prediction Analysis
    # ══════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-header">تحليل التوقعات مقابل القيم الحقيقية</div>',
                    unsafe_allow_html=True)

        sel = st.selectbox("اختر النموذج لعرض التحليل", list(results.keys()), key="tab4_sel")
        r   = results[sel]

        col1, col2 = st.columns(2)
        with col1:
            # Actual vs Predicted scatter
            fig_sc = px.scatter(x=r["y_val"], y=r["val_pred"],
                                labels={"x": "القيمة الحقيقية", "y": "القيمة المتوقعة"},
                                title=f"{sel} — Actual vs Predicted",
                                color_discrete_sequence=[ALGO_INFO[sel]["color"]],
                                opacity=0.6)
            max_val = max(r["y_val"].max(), r["val_pred"].max())
            fig_sc.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                             line=dict(color="white", dash="dash"))
            fig_sc.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sc, use_container_width=True)

        with col2:
            # Residuals
            residuals = r["y_val"] - r["val_pred"]
            fig_res = px.histogram(x=residuals, nbins=40, title="توزيع الأخطاء (Residuals)",
                                   color_discrete_sequence=[ALGO_INFO[sel]["color"]])
            fig_res.add_vline(x=0, line_dash="dash", line_color="white")
            fig_res.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  xaxis_title="Error (Actual − Predicted)",
                                  yaxis_title="Count")
            st.plotly_chart(fig_res, use_container_width=True)

        # Feature importance (Random Forest & XGBoost only)
        if sel in ("Random Forest", "XGBoost"):
            st.markdown('<div class="section-header">أهمية المتغيرات (Feature Importance)</div>',
                        unsafe_allow_html=True)
            importances = r["model"].feature_importances_
            fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
            fi_df = fi_df.sort_values("Importance", ascending=True)
            fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                            title=f"{sel} — Feature Importance",
                            color="Importance",
                            color_continuous_scale=["#2d3556", ALGO_INFO[sel]["color"]])
            fig_fi.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                                 coloraxis_showscale=False)
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info(f"ℹ️ Feature Importance متاحة فقط لـ Random Forest و XGBoost."
                    f" اختر أحدهما لرؤية أهمية المتغيرات.")

    # ── Footer
    st.markdown("""
    <hr style="border-color:#2d3556; margin-top:40px;">
    <p style="text-align:center; color:#4a5568; font-size:0.8rem;">
        🔥 Calories Burnt Prediction Dashboard | 
        Algorithms: Linear Regression · Lasso · Ridge · Random Forest · XGBoost
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
