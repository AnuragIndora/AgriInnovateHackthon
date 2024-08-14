import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Load models
regressor = joblib.load('regressor_model.pkl')
classifier = joblib.load('classifier_model.pkl')
model_cls = tf.keras.models.load_model('deep_learning_classifier_model.h5')
label_encoder_classes = joblib.load('label_encoder_classes.pkl')

# Load test data for evaluation
df = pd.read_csv('combined_data_june_2024.csv')
df = df.drop(columns=['timestamp', 'date_x', 'date_y'])
label_encoder = joblib.load('label_encoder_classes.pkl')

numerical_features = ['soil_moisture', 'soil_nutrients', 'soil_ph', 'temperature_x',
                       'humidity_x', 'temperature_y', 'humidity_y', 'rainfall',
                       'price_per_unit', 'yield_per_hectare', 'water_usage',
                       'carbon_emission']

imputer = SimpleImputer(strategy='median')
df[numerical_features] = imputer.fit_transform(df[numerical_features])

X = df[numerical_features]
y_classification = df['crop']
y_regression = df['yield_per_hectare']

# Split the data
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_classification, test_size=0.2, random_state=42)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Crop Yield Prediction'),

    dcc.Input(id='soil_moisture', type='number', placeholder='Soil Moisture'),
    dcc.Input(id='soil_nutrients', type='number', placeholder='Soil Nutrients'),
    dcc.Input(id='soil_ph', type='number', placeholder='Soil pH'),
    dcc.Input(id='temperature_x', type='number', placeholder='Temperature X'),
    dcc.Input(id='humidity_x', type='number', placeholder='Humidity X'),
    dcc.Input(id='temperature_y', type='number', placeholder='Temperature Y'),
    dcc.Input(id='humidity_y', type='number', placeholder='Humidity Y'),
    dcc.Input(id='rainfall', type='number', placeholder='Rainfall'),
    dcc.Input(id='price_per_unit', type='number', placeholder='Price per Unit'),
    dcc.Input(id='yield_per_hectare', type='number', placeholder='Yield per Hectare'),
    dcc.Input(id='water_usage', type='number', placeholder='Water Usage'),
    dcc.Input(id='carbon_emission', type='number', placeholder='Carbon Emission'),

    html.Button('Predict', id='predict-button', n_clicks=0),
    html.Div(id='prediction-output'),

    html.H2('Regression Metrics'),
    dcc.Graph(id='regression-metrics'),

    html.H2('Classification Metrics'),
    dcc.Graph(id='classification-metrics')
])

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [dash.dependencies.State('soil_moisture', 'value'),
     dash.dependencies.State('soil_nutrients', 'value'),
     dash.dependencies.State('soil_ph', 'value'),
     dash.dependencies.State('temperature_x', 'value'),
     dash.dependencies.State('humidity_x', 'value'),
     dash.dependencies.State('temperature_y', 'value'),
     dash.dependencies.State('humidity_y', 'value'),
     dash.dependencies.State('rainfall', 'value'),
     dash.dependencies.State('price_per_unit', 'value'),
     dash.dependencies.State('yield_per_hectare', 'value'),
     dash.dependencies.State('water_usage', 'value'),
     dash.dependencies.State('carbon_emission', 'value')]
)
def update_output(n_clicks, soil_moisture, soil_nutrients, soil_ph, temperature_x,
                  humidity_x, temperature_y, humidity_y, rainfall, price_per_unit,
                  yield_per_hectare, water_usage, carbon_emission):

    if n_clicks > 0:
        input_data = pd.DataFrame([{
            'soil_moisture': soil_moisture,
            'soil_nutrients': soil_nutrients,
            'soil_ph': soil_ph,
            'temperature_x': temperature_x,
            'humidity_x': humidity_x,
            'temperature_y': temperature_y,
            'humidity_y': humidity_y,
            'rainfall': rainfall,
            'price_per_unit': price_per_unit,
            'yield_per_hectare': yield_per_hectare,
            'water_usage': water_usage,
            'carbon_emission': carbon_emission
        }])

        # Predict using traditional model
        pred_class = classifier.predict(input_data)[0]
        pred_class_label = label_encoder_classes[pred_class]

        # Predict using deep learning model
        input_data_dl = np.array(input_data)
        pred_class_dl = np.argmax(model_cls.predict(input_data_dl), axis=-1)[0]
        pred_class_label_dl = label_encoder_classes[pred_class_dl]

        return f"Traditional Model Prediction: {pred_class_label}, Deep Learning Model Prediction: {pred_class_label_dl}"

    return "Enter values and click 'Predict'"

@app.callback(
    Output('regression-metrics', 'figure'),
    [Input('predict-button', 'n_clicks')]
)
def update_regression_metrics(n_clicks):
    if n_clicks > 0:
        y_pred_reg = regressor.predict(X_test)
        mse = mean_squared_error(y_test_reg, y_pred_reg)
        r2 = r2_score(y_test_reg, y_pred_reg)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Mean Squared Error', 'R^2 Score'],
            y=[mse, r2],
            marker_color=['#FF5733', '#33FF57']
        ))

        fig.update_layout(title='Regression Metrics', xaxis_title='Metric', yaxis_title='Value')
        return fig

    return dash.no_update

@app.callback(
    Output('classification-metrics', 'figure'),
    [Input('predict-button', 'n_clicks')]
)
def update_classification_metrics(n_clicks):
    if n_clicks > 0:
        y_pred_cls = classifier.predict(X_test_cls)
        accuracy = accuracy_score(y_test_cls, y_pred_cls)
        report = classification_report(y_test_cls, y_pred_cls, output_dict=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Accuracy'],
            y=[accuracy],
            marker_color=['#FF5733']
        ))

        for label, metrics in report.items():
            if label != 'accuracy':
                fig.add_trace(go.Bar(
                    x=[f'{label} Precision', f'{label} Recall', f'{label} F1-Score'],
                    y=[metrics['precision'], metrics['recall'], metrics['f1-score']],
                    name=label
                ))

        fig.update_layout(title='Classification Metrics', xaxis_title='Metric', yaxis_title='Value')
        return fig

    return dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
