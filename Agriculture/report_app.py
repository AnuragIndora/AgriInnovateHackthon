import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import joblib
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Initialize the Dash app
app = dash.Dash(__name__)

# Load the models and encoders
regressor = joblib.load('regressor_model.pkl')
classifier = joblib.load('classifier_model.pkl')
model_reg_history = joblib.load('model_reg_history.pkl')  # History of deep learning regression
model_cls_history = joblib.load('model_cls_history.pkl')  # History of deep learning classification

# Load and preprocess data
df = pd.read_csv('combined_data_june_2024.csv')
df = df.drop(columns=['timestamp', 'date_x', 'date_y'])

# Prepare data for predictions
X = df.drop(columns=['yield_per_hectare', 'crop'])
y_regression = df['yield_per_hectare']
y_classification = df['crop']

# Encode classification labels
label_encoder = LabelEncoder()
y_classification_encoded = label_encoder.fit_transform(y_classification)

# Make predictions
y_pred_reg = regressor.predict(X)
y_pred_cls_encoded = classifier.predict(X)

# Decode predictions for classification
y_pred_cls = label_encoder.inverse_transform(y_pred_cls_encoded)

# Compute performance metrics
mse = mean_squared_error(y_regression, y_pred_reg)
r2 = r2_score(y_regression, y_pred_reg)
accuracy = accuracy_score(y_classification, y_pred_cls)
report = classification_report(y_classification, y_pred_cls, output_dict=True)

# Get training history data
def get_history(history_dict, key):
    if key in history_dict:
        return history_dict[key]
    return []

# Create Dash layout
app.layout = html.Div([
    html.H1('Agriculture Model Dashboard'),

    html.Div([
        html.H2('Regression Model Performance'),
        dcc.Graph(
            figure={
                'data': [
                    go.Bar(
                        x=['MSE', 'R2 Score'],
                        y=[mse, r2],
                        marker={'color': 'blue'}
                    )
                ],
                'layout': go.Layout(
                    title='Regression Model Metrics',
                    xaxis={'title': 'Metric'},
                    yaxis={'title': 'Value'}
                )
            }
        )
    ]),

    html.Div([
        html.H2('Classification Model Performance'),
        dcc.Graph(
            figure={
                'data': [
                    go.Bar(
                        x=['Accuracy'],
                        y=[accuracy],
                        marker={'color': 'green'}
                    )
                ],
                'layout': go.Layout(
                    title='Classification Model Accuracy',
                    xaxis={'title': 'Metric'},
                    yaxis={'title': 'Value'}
                )
            }
        ),
        html.H3('Classification Report'),
        html.Pre(f'{report}', style={'white-space': 'pre-wrap'})
    ]),

    html.Div([
        html.H2('Regression Model Training History'),
        dcc.Graph(
            figure={
                'data': [
                    go.Scatter(
                        x=list(range(len(get_history(model_reg_history, 'loss')))),
                        y=get_history(model_reg_history, 'loss'),
                        mode='lines+markers',
                        name='Loss'
                    ),
                    go.Scatter(
                        x=list(range(len(get_history(model_reg_history, 'val_loss')))),
                        y=get_history(model_reg_history, 'val_loss'),
                        mode='lines+markers',
                        name='Validation Loss'
                    )
                ],
                'layout': go.Layout(
                    title='Deep Learning Regression Model Training History',
                    xaxis={'title': 'Epoch'},
                    yaxis={'title': 'Value'}
                )
            }
        )
    ]),

    html.Div([
        html.H2('Classification Model Training History'),
        dcc.Graph(
            figure={
                'data': [
                    go.Scatter(
                        x=list(range(len(get_history(model_cls_history, 'loss')))),
                        y=get_history(model_cls_history, 'loss'),
                        mode='lines+markers',
                        name='Loss'
                    ),
                    go.Scatter(
                        x=list(range(len(get_history(model_cls_history, 'val_loss')))),
                        y=get_history(model_cls_history, 'val_loss'),
                        mode='lines+markers',
                        name='Validation Loss'
                    )
                ],
                'layout': go.Layout(
                    title='Deep Learning Classification Model Training History',
                    xaxis={'title': 'Epoch'},
                    yaxis={'title': 'Value'}
                )
            }
        )
    ])
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
