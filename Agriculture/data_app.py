import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd

# Load datasets
soil_health_df = pd.read_csv('soil_health_data_june_2024.csv')
weather_df = pd.read_csv('weather_data_june_2024.csv')
market_prices_df = pd.read_csv('market_prices_june_2024.csv')
sensor_data_df = pd.read_csv('iot_sensor_data_june_2024.csv')
crop_yield_df = pd.read_csv('crop_yield_data_june_2024.csv')
user_interaction_df = pd.read_csv('user_interaction_data_june_2024.csv')
sustainability_metrics_df = pd.read_csv('sustainability_metrics_june_2024.csv')
community_data_df = pd.read_csv('community_data_june_2024.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Agriculture Dashboard Prototype"),
    
    html.Div([
        html.H2("Soil Health Data"),
        dcc.Graph(
            figure={
                'data': [
                    go.Scatter(
                        x=soil_health_df['timestamp'],
                        y=soil_health_df['soil_moisture'],
                        mode='lines+markers',
                        name='Soil Moisture'
                    ),
                    go.Scatter(
                        x=soil_health_df['timestamp'],
                        y=soil_health_df['soil_nutrients'],
                        mode='lines+markers',
                        name='Soil Nutrients'
                    ),
                    go.Scatter(
                        x=soil_health_df['timestamp'],
                        y=soil_health_df['soil_ph'],
                        mode='lines+markers',
                        name='Soil pH'
                    )
                ],
                'layout': go.Layout(
                    title='Soil Health Over Time',
                    xaxis={'title': 'Timestamp'},
                    yaxis={'title': 'Values'}
                )
            }
        )
    ]),

    html.Div([
        html.H2("Weather Data"),
        dcc.Graph(
            figure={
                'data': [
                    go.Scatter(
                        x=weather_df['date'],
                        y=weather_df['temperature'],
                        mode='lines+markers',
                        name='Temperature'
                    ),
                    go.Scatter(
                        x=weather_df['date'],
                        y=weather_df['humidity'],
                        mode='lines+markers',
                        name='Humidity'
                    ),
                    go.Scatter(
                        x=weather_df['date'],
                        y=weather_df['rainfall'],
                        mode='lines+markers',
                        name='Rainfall'
                    )
                ],
                'layout': go.Layout(
                    title='Weather Conditions Over Time',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Values'}
                )
            }
        )
    ]),

    html.Div([
        html.H2("Market Prices"),
        dcc.Graph(
            figure={
                'data': [
                    go.Bar(
                        x=market_prices_df['date'],
                        y=market_prices_df['price_per_unit'],
                        name='Price per Unit'
                    )
                ],
                'layout': go.Layout(
                    title='Market Prices Over Time',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Price per Unit'}
                )
            }
        )
    ]),

    html.Div([
        html.H2("Sensor Data"),
        dcc.Graph(
            figure={
                'data': [
                    go.Scatter(
                        x=sensor_data_df['timestamp'],
                        y=sensor_data_df['temperature'],
                        mode='lines+markers',
                        name='Temperature'
                    ),
                    go.Scatter(
                        x=sensor_data_df['timestamp'],
                        y=sensor_data_df['humidity'],
                        mode='lines+markers',
                        name='Humidity'
                    )
                ],
                'layout': go.Layout(
                    title='Sensor Data Over Time',
                    xaxis={'title': 'Timestamp'},
                    yaxis={'title': 'Values'}
                )
            }
        )
    ]),

    html.Div([
        html.H2("Crop Yield Data"),
        dcc.Graph(
            figure={
                'data': [
                    go.Bar(
                        x=crop_yield_df['crop'],
                        y=crop_yield_df['yield_per_hectare'],
                        name='Yield per Hectare'
                    )
                ],
                'layout': go.Layout(
                    title='Crop Yield',
                    xaxis={'title': 'Crop'},
                    yaxis={'title': 'Yield per Hectare'}
                )
            }
        )
    ]),

    html.Div([
        html.H2("User Interaction Data"),
        dcc.Graph(
            figure={
                'data': [
                    go.Bar(
                        x=user_interaction_df['user_id'],
                        y=user_interaction_df['interaction_type'].astype(str).value_counts(),
                        name='Interaction Types'
                    )
                ],
                'layout': go.Layout(
                    title='User Interaction Types',
                    xaxis={'title': 'User ID'},
                    yaxis={'title': 'Count'}
                )
            }
        )
    ]),

    html.Div([
        html.H2("Sustainability Metrics"),
        dcc.Graph(
            figure={
                'data': [
                    go.Scatter(
                        x=sustainability_metrics_df['timestamp'],
                        y=sustainability_metrics_df['water_usage'],
                        mode='lines+markers',
                        name='Water Usage'
                    ),
                    go.Scatter(
                        x=sustainability_metrics_df['timestamp'],
                        y=sustainability_metrics_df['carbon_emission'],
                        mode='lines+markers',
                        name='Carbon Emission'
                    )
                ],
                'layout': go.Layout(
                    title='Sustainability Metrics Over Time',
                    xaxis={'title': 'Timestamp'},
                    yaxis={'title': 'Values'}
                )
            }
        )
    ]),

    html.Div([
        html.H2("Community Data"),
        dcc.Graph(
            figure={
                'data': [
                    go.Bar(
                        x=community_data_df['user_id'],
                        y=community_data_df['post_id'].astype(str).value_counts(),
                        name='Posts per User'
                    )
                ],
                'layout': go.Layout(
                    title='Community Posts by User',
                    xaxis={'title': 'User ID'},
                    yaxis={'title': 'Number of Posts'}
                )
            }
        )
    ])
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
