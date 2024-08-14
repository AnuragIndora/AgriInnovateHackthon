import pandas as pd

# Load the datasets
soil_health_df = pd.read_csv('soil_health_data_june_2024.csv')
weather_df = pd.read_csv('weather_data_june_2024.csv')
market_prices_df = pd.read_csv('market_prices_june_2024.csv')
sensor_data_df = pd.read_csv('iot_sensor_data_june_2024.csv')
crop_yield_df = pd.read_csv('crop_yield_data_june_2024.csv')
user_interaction_df = pd.read_csv('user_interaction_data_june_2024.csv')
sustainability_metrics_df = pd.read_csv('sustainability_metrics_june_2024.csv')
community_data_df = pd.read_csv('community_data_june_2024.csv')

# Print column names for debugging
print("Soil Health Data Columns:", soil_health_df.columns)
print("Weather Data Columns:", weather_df.columns)
print("Market Prices Data Columns:", market_prices_df.columns)
print("Sensor Data Columns:", sensor_data_df.columns)
print("Crop Yield Data Columns:", crop_yield_df.columns)
print("User Interaction Data Columns:", user_interaction_df.columns)
print("Sustainability Metrics Data Columns:", sustainability_metrics_df.columns)
print("Community Data Columns:", community_data_df.columns)

# Merge datasets step by step

# Merge soil health and sensor data on 'timestamp'
combined_df = pd.merge(soil_health_df, sensor_data_df, left_on='timestamp', right_on='timestamp', how='outer')

# Merge with weather data on 'timestamp' or 'date'
if 'date' in weather_df.columns:
    combined_df = pd.merge(combined_df, weather_df, left_on='timestamp', right_on='date', how='outer')
else:
    combined_df = pd.merge(combined_df, weather_df, left_on='timestamp', right_on='timestamp', how='outer')

# Merge with market prices on 'timestamp' or 'date'
if 'date' in market_prices_df.columns:
    combined_df = pd.merge(combined_df, market_prices_df, left_on='timestamp', right_on='date', how='left')
else:
    combined_df = pd.merge(combined_df, market_prices_df, left_on='timestamp', right_on='timestamp', how='left')

# Merge with crop yield data on 'crop'
if 'crop' in combined_df.columns and 'crop' in crop_yield_df.columns:
    combined_df = pd.merge(combined_df, crop_yield_df, on='crop', how='left')

# Merge with user interaction data on 'user_id'
if 'user_id' in combined_df.columns and 'user_id' in user_interaction_df.columns:
    combined_df = pd.merge(combined_df, user_interaction_df, on='user_id', how='left')

# Merge with sustainability metrics on 'timestamp'
if 'timestamp' in combined_df.columns and 'timestamp' in sustainability_metrics_df.columns:
    combined_df = pd.merge(combined_df, sustainability_metrics_df, on='timestamp', how='left')

# Merge with community data on 'user_id'
if 'user_id' in combined_df.columns and 'user_id' in community_data_df.columns:
    combined_df = pd.merge(combined_df, community_data_df, on='user_id', how='left')

# Save the combined dataset to a CSV file
combined_df.to_csv('combined_data_june_2024.csv', index=False)

print("Datasets combined and saved to 'combined_data_june_2024.csv'")
