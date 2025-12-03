import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
import yaml

def preprocess():
    # Load data
    df = pd.read_csv('data/cpu_usage.csv')
    
    # Separate features and target
    X = df.drop('cpu_usage', axis=1)
    y = df['cpu_usage']
    
    # One-hot encode categorical variables
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_encoded = pd.DataFrame(encoder.fit_transform(X[['controller_kind']]))
    X_encoded.columns = encoder.get_feature_names_out(['controller_kind'])
    
    # Concatenate encoded features and drop original categorical column
    X = pd.concat([X.drop('controller_kind', axis=1), X_encoded], axis=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    print("Preprocessing complete. Data saved to data/processed/")

if __name__ == "__main__":
    preprocess()
