import pandas as pd
import pickle
import json
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Note: The user mentioned Logistic Regression, but this is a regression problem (predicting CPU usage).
# I will use Linear Regression instead of Logistic Regression for the baseline.
# I will use SVR instead of SVM (Classifier).
# I will use RandomForestRegressor.

def train():
    # Load processed data
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    models = {
        'LinearRegression': LinearRegression(),
        'SVR': SVR(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = -float('inf')
    metrics = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        metrics[name] = {
            'R2': r2,
            'MAE': mae,
            'RMSE': rmse
        }
        
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_model_name = name
            
    print(f"Best model: {best_model_name} with R2: {best_score}")
    
    # Save metrics
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Save best model
    os.makedirs('models', exist_ok=True)
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
        
    print("Training complete. Best model saved to models/model.pkl")

if __name__ == "__main__":
    train()
