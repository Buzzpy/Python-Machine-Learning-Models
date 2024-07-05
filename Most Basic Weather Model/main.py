from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Load the dataset
weather_data = pd.read_csv('weather_data.csv')

# Preprocess the data
label_encoder = LabelEncoder()
weather_data['Condition'] = label_encoder.fit_transform(weather_data['Condition'])

# Define features and target
X = weather_data[['Temperature', 'Humidity']]
y = weather_data['Condition']

# Initialize the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X, y)

# Predict the weather for the next week
future_dates = pd.date_range(start='2023-06-29', end='2023-07-05', freq='D')
future_temperature = np.random.randint(60, 100, size=len(future_dates))
future_humidity = np.random.randint(30, 80, size=len(future_dates))

future_data = pd.DataFrame({
    'Temperature': future_temperature,
    'Humidity': future_humidity
})

future_predictions = clf.predict(future_data)
future_conditions = label_encoder.inverse_transform(future_predictions)

future_weather = pd.DataFrame({
    'Date': future_dates,
    'Temperature': future_temperature,
    'Humidity': future_humidity,
    'Predicted_Condition': future_conditions
})

print("\nPredicted weather for the next week:")
print(future_weather)
