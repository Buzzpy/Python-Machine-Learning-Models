import pandas as pd
import numpy as np

# Define the date range
date_range = pd.date_range(start='2023-06-15', end='2023-06-28', freq='D')

# Generate sample data
np.random.seed(42)
temperature = np.random.randint(60, 100, size=len(date_range))  # Random temperatures between 60 and 100
humidity = np.random.randint(30, 80, size=len(date_range))      # Random humidity between 30 and 80
conditions = np.random.choice(['sunny', 'rainy', 'cloudy'], size=len(date_range))  # Random weather conditions

# Create a DataFrame
weather_data = pd.DataFrame({
    'Date': date_range,
    'Temperature': temperature,
    'Humidity': humidity,
    'Condition': conditions
})

# Save to CSV
weather_data.to_csv('weather_data.csv', index=False)

print("Sample weather data for past two weeks:")
print(weather_data)

