# retrain_models.py

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# 1. Load your dataset
# (replace this with your real dataset path)
data = pd.read_csv('Adidas US Sales Datasets.csv')

# 2. Basic preprocessing
# Assuming you have columns like 'Retail Price', 'Sales', 'Region', etc.

# For Random Forest Regressor
features = ['Retail Price', 'Product ID', 'Units Sold']  # adjust based on your data
target = 'Sales'

X = data[features]
y = data[target]

# 3. Train Random Forest Regressor
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X, y)

# 4. Save Random Forest model
os.makedirs('models', exist_ok=True)
joblib.dump(random_forest_model, 'models/random_forest_model.pkl')
print('Random Forest model saved!')

# For Naive Bayes (we'll make a simple classifier: e.g., predicting if "Units Sold" > threshold)
data['High_Sales'] = (data['Units Sold'] > data['Units Sold'].median()).astype(int)

X_nb = data[['Retail Price']]  # you can expand features
y_nb = data['High_Sales']

# 5. Train Naive Bayes Classifier
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_nb, y_nb)

# 6. Save Naive Bayes model
joblib.dump(naive_bayes_model, 'models/naive_bayes_model.pkl')
print('Naive Bayes model saved!')
