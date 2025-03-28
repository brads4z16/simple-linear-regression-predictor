import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle
import os
from config import DATA_PATH, MODEL_FILE, SHOW_PLOT

class ChicagoTaxiModel:
    def __init__(self):
        self.model = LinearRegression()
        self.label_encoders = {}
        
    def save_model(self):
        """
        Save the trained model and label encoders to a file
        """
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders
        }
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {MODEL_FILE}")
    
    def load_data(self, file_path):
        """
        Load and preprocess the Chicago taxi data
        
        Args:
            file_path (str): Path to the CSV file containing taxi data
            
        Returns:
            pd.DataFrame: Processed dataset
        """
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def prepare_features(self, df):
        """
        Prepare features for the model
        
        Args:
            df (pd.DataFrame): Raw taxi data
            
        Returns:
            tuple: (X, y) where X is feature matrix and y is target variable
        """
        # Select features and target
        features = ['TRIP_MILES', 'TRIP_SECONDS', 'COMPANY', 'TIP_RATE']
        target = 'FARE'
        
        # Create a copy of the data to avoid modifying the original
        X = df[features].copy()
        y = df[target]
        
        # Encode categorical variables
        categorical_features = ['COMPANY']
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
            X[feature] = self.label_encoders[feature].fit_transform(X[feature])
        
        return X, y
    
    def train(self, X, y):
        """
        Train the linear regression model
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target variable
        """
        self.model.fit(X, y)
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Args:
            X (np.array): Feature matrix
            
        Returns:
            np.array: Predicted values
        """
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        Evaluate the model performance
        
        Args:
            X (np.array): Feature matrix
            y (np.array): True values
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2
        }

    def plot_predictions(self, y_true, y_pred):
        """
        Plot the predictions against the true values
        
        Args:
            y_true (np.array): True values
            y_pred (np.array): Predicted values
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('True vs Predicted Values')
        plt.show()

def main():
    # Initialize model
    model = ChicagoTaxiModel()
    
    # Load data
    df = model.load_data(DATA_PATH)
    X, y = model.prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # Train and evaluate new model
    print("\nTraining new model...")
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    print("Model Performance:", metrics)
    
    # Save the new model
    model.save_model()
    
    # Show plot if configured
    if SHOW_PLOT:
        y_pred = model.predict(X_test)
        model.plot_predictions(y_test, y_pred)

if __name__ == "__main__":
    main() 