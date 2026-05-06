from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Train all models on startup ──────────────────────────────────────────────
df_exercise = pd.read_csv('exercise.csv')
df_calories = pd.read_csv('calories.csv')
df = pd.merge(df_exercise, df_calories, on='User_ID')

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])   # male=1, female=0

# Drop columns not used as features (per notebook)
df.drop(['Weight', 'Duration'], axis=1, inplace=True)

features = df.drop(['User_ID', 'Calories'], axis=1)   # Gender Age Height Heart_Rate Body_Temp
target = df['Calories'].values

X_train, X_val, Y_train, Y_val = train_test_split(
    features, target, test_size=0.1, random_state=22
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)

MODELS = {
    "Linear Regression":   LinearRegression(),
    "Gradient Boosting":   GradientBoostingRegressor(),
    "Lasso":               Lasso(),
    "Random Forest":       RandomForestRegressor(),
    "Ridge":               Ridge(),
}

model_metrics = {}
for name, mdl in MODELS.items():
    mdl.fit(X_train_s, Y_train)
    val_preds = mdl.predict(X_val_s)
    model_metrics[name] = round(float(mean_absolute_error(Y_val, val_preds)), 3)

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', model_metrics=model_metrics)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    gender     = 1 if data['gender'].lower() == 'male' else 0
    age        = float(data['age'])
    height     = float(data['height'])
    heart_rate = float(data['heart_rate'])
    body_temp  = float(data['body_temp'])

    input_arr = np.array([[gender, age, height, heart_rate, body_temp]])
    input_scaled = scaler.transform(input_arr)

    results = {}
    for name, mdl in MODELS.items():
        pred = mdl.predict(input_scaled)[0]
        results[name] = round(float(pred), 1)

    return jsonify({'predictions': results, 'metrics': model_metrics})


if __name__ == '__main__':
    app.run(debug=True)
