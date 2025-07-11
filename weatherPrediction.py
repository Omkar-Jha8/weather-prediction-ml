import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import joblib
import requests
from datetime import datetime


API_KEY = "Your API Key"
CITY_ID = "1273294"  # Delhi
LAT = 28.6667
LON = 77.2167
url_forecast = f"https://api.openweathermap.org/data/2.5/forecast?q=Delhi&appid={API_KEY}&units=metric"

response = requests.get(url_forecast)
forecast_data = response.json()

def extract_features_from_forecast(forecast_data, target_date="2025-07-11"):
    features = {
        "MinTemp": 0,
        "MaxTemp": 0,
        "Rainfall": 0,
        "Sunshine": 0,
        "WindSpeed9am": 0,
        "WindSpeed3pm": 0,
        "Humidity9am": 0,
        "Humidity3pm": 0,
        "Pressure9am": 0,
        "Pressure3pm": 0,
        "Cloud9am": 0,
        "Cloud3pm": 0,
        "Temp9am": 0,
        "Temp3pm": 0,
        "RainToday": 0
    }

    rain_today = 0
    min_temp = float("inf")
    max_temp = float("-inf")
    rainfall_total = 0

    for entry in forecast_data["list"]:
        dt_txt = entry["dt_txt"]
        if not dt_txt.startswith(target_date):
            continue

        main = entry["main"]
        clouds = entry["clouds"]["all"]
        wind = entry["wind"]["speed"]
        rain = entry.get("rain", {}).get("3h", 0)

        time = dt_txt.split()[1]

        # Min & Max temp
        min_temp = min(min_temp, main["temp_min"])
        max_temp = max(max_temp, main["temp_max"])
        rainfall_total += rain
        if rain > 0:
            rain_today = 1

        if time == "09:00:00":
            features["WindSpeed9am"] = wind
            features["Humidity9am"] = main["humidity"]
            features["Pressure9am"] = main["pressure"]
            features["Cloud9am"] = clouds
            features["Temp9am"] = main["temp"]
        elif time == "15:00:00":
            features["WindSpeed3pm"] = wind
            features["Humidity3pm"] = main["humidity"]
            features["Pressure3pm"] = main["pressure"]
            features["Cloud3pm"] = clouds
            features["Temp3pm"] = main["temp"]

    # Ensure temp values are not infinity
    if min_temp == float("inf"):
        min_temp = 0
    if max_temp == float("-inf"):
        max_temp = 0

    features["MinTemp"] = min_temp
    features["MaxTemp"] = max_temp
    features["Rainfall"] = rainfall_total
    features["RainToday"] = rain_today

    cloud_avg = (features["Cloud9am"] + features["Cloud3pm"]) / 2
    features["Sunshine"] = round(12 * (1 - cloud_avg / 100), 2)
    print("🌦️ Features passed to model for prediction:")
    for key, value in features.items():
        print(f"{key}: {value}")
    return np.array([[ 
        features["MinTemp"],
        features["MaxTemp"],
        features["Rainfall"],
        features["Sunshine"],
        features["WindSpeed9am"],
        features["WindSpeed3pm"],
        features["Humidity9am"],
        features["Humidity3pm"],
        features["Pressure9am"],
        features["Pressure3pm"],
        features["Cloud9am"],
        features["Cloud3pm"],
        features["Temp9am"],
        features["Temp3pm"],
        features["RainToday"]
    ]])
    



df = pd.read_csv("weather.csv")

# Show the shape and first few rows
# Drop missing rows
df.dropna(inplace=True)

df['RainToday'] = df['RainToday'].map({'Yes':1,'No':0})
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes':1,'No':0})


# Select input features
X = df[[ "MinTemp", "MaxTemp", "Rainfall", "Sunshine", 
         "WindSpeed9am", "WindSpeed3pm", 
         "Humidity9am", "Humidity3pm", 
         "Pressure9am", "Pressure3pm", 
         "Cloud9am", "Cloud3pm", 
         "Temp9am", "Temp3pm", "RainToday" ]]

# Target: actual temperature
y = df["RainTomorrow"]



# Split the data: 80% train, 20% test
# X_train, X_test, y_train,y_test = train_test_split(
#     X,y,test_size=0.2,random_state=42
# )

# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)
# joblib.dump(model, "RainPred.joblib")

model = joblib.load("RainPred.joblib")

input_data = extract_features_from_forecast(forecast_data, target_date="2025-07-11")
prediction = model.predict(input_data)
if prediction[0] == 1:
    print("Prediction: 🌧️ It WILL rain tomorrow.")
    
else:
    print("Prediction: ☀️ It will NOT rain tomorrow.")
