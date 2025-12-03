import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import os

def evaluate():
    # Load data
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    # Load model
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    metrics = {
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse
    }
    
    print(f"Evaluation Metrics: {metrics}")
    
    # Save metrics
    with open('evaluation.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Generate plots
    os.makedirs('plots', exist_ok=True)
    
    # Actual vs Predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual CPU Usage')
    plt.ylabel('Predicted CPU Usage')
    plt.title('Actual vs Predicted CPU Usage')
    plt.savefig('plots/actual_vs_predicted.png')
    plt.close()
    
    print("Evaluation complete. Metrics saved to evaluation.json and plots to plots/")

if __name__ == "__main__":
    evaluate()
