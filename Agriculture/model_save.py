import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load and preprocess the dataset
df = pd.read_csv('combined_data_june_2024.csv')

# Drop unnecessary columns
df = df.drop(columns=['timestamp', 'date_x', 'date_y'])

# Encode 'crop' column
label_encoder = LabelEncoder()
df['crop'] = label_encoder.fit_transform(df['crop'])

# Define numerical features
numerical_features = ['soil_moisture', 'soil_nutrients', 'soil_ph', 'temperature_x',
                       'humidity_x', 'temperature_y', 'humidity_y', 'rainfall',
                       'price_per_unit', 'yield_per_hectare', 'water_usage',
                       'carbon_emission']

# Ensure all features are numerical
imputer = SimpleImputer(strategy='median')
df[numerical_features] = imputer.fit_transform(df[numerical_features])

# Separate features and targets
X = df.drop(columns=['yield_per_hectare', 'crop'])
y_regression = df['yield_per_hectare']
y_classification = df['crop']

# Split the data for regression and classification
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_classification, test_size=0.2, random_state=42)

# Train and save the regression model
print("Training Regression Model...")
regressor = LinearRegression()
regressor.fit(X_train, y_train_reg)
joblib.dump(regressor, 'regressor_model.pkl')

# Train and save the deep learning regression model
print("Training Deep Learning Regression Model...")
model_reg = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
model_reg.compile(optimizer='adam', loss='mean_squared_error')
history_reg = model_reg.fit(X_train, y_train_reg, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
model_reg.save('deep_learning_regressor_model.h5')
joblib.dump(history_reg.history, 'model_reg_history.pkl')

# Train and save the classification model
print("Training Classification Model...")
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_cls, y_train_cls)
joblib.dump(classifier, 'classifier_model.pkl')

# Train and save the deep learning classification model
print("Training Deep Learning Classification Model...")
model_cls = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_cls.shape[1],)),
    Dense(32, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])
model_cls.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_cls = model_cls.fit(X_train_cls, y_train_cls, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
model_cls.save('deep_learning_classifier_model.h5')
joblib.dump(history_cls.history, 'model_cls_history.pkl')

# Save the label encoder classes
print("Saving Label Encoder Classes...")
joblib.dump(label_encoder.classes_, 'label_encoder_classes.pkl')

print("Models and training histories have been saved.")
