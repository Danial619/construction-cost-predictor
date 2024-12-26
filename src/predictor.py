# Construction Cost Predictor
# Author: [Your Name]
# Description: A machine learning model to predict construction project costs
# GitHub Portfolio Project

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

class ConstructionCostPredictor:
    """
    A class to predict construction project costs using machine learning.
    """
    
    def __init__(self):
        """Initialize the predictor"""
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def create_sample_data(self, n_samples=500):
        """
        Create sample construction project data
        
        Parameters:
        n_samples (int): Number of samples to generate
        
        Returns:
        pandas DataFrame: Generated construction data
        """
        data = {
            'project_size': np.random.randint(100, 10000, n_samples),
            'material_cost': np.random.uniform(100, 1000, n_samples),
            'labor_hours': np.random.randint(100, 5000, n_samples)
        }
        
        df = pd.DataFrame(data)
        # Calculate total cost based on a simplified formula
        df['total_cost'] = (df['material_cost'] * df['project_size'] / 100 + 
                           df['labor_hours'] * 50)
        
        return df
    
    def prepare_data(self, df):
        """
        Prepare data for training
        
        Parameters:
        df (pandas DataFrame): Input data
        
        Returns:
        tuple: X (features) and y (target) data
        """
        X = df[['project_size', 'material_cost', 'labor_hours']]
        y = df['total_cost']
        return X, y
    
    def train_model(self, X, y):
        """
        Train the prediction model
        
        Parameters:
        X (array-like): Feature data
        y (array-like): Target data
        """
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        return X_test, y_test, y_pred
    
    def plot_results(self, y_test, y_pred):
        """
        Visualize the prediction results
        
        Parameters:
        y_test (array-like): Actual values
        y_pred (array-like): Predicted values
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', lw=2)
        plt.xlabel('Actual Cost (£)')
        plt.ylabel('Predicted Cost (£)')
        plt.title('Construction Cost Prediction Results')
        plt.tight_layout()
        plt.show()
        
    def plot_feature_importance(self, feature_names):
        """
        Plot feature importance
        
        Parameters:
        feature_names (list): Names of features
        """
        importance = self.model.feature_importances_
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance, y=feature_names)
        plt.title('Feature Importance in Cost Prediction')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the construction cost predictor"""
    
    # Initialize predictor
    predictor = ConstructionCostPredictor()
    
    # Create sample data
    print("Generating construction project data...")
    df = predictor.create_sample_data()
    
    # Prepare data
    print("Preparing data for training...")
    X, y = predictor.prepare_data(df)
    
    # Train model and get predictions
    print("Training the model...")
    X_test, y_test, y_pred = predictor.train_model(X, y)
    
    # Plot results
    print("Generating visualizations...")
    predictor.plot_results(y_test, y_pred)
    predictor.plot_feature_importance(['Project Size', 
                                     'Material Cost', 
                                     'Labor Hours'])
    
    # Calculate and display model accuracy
    accuracy = np.mean(np.abs(y_pred - y_test) / y_test) * 100
    print(f"\nModel Performance:")
    print(f"Average Prediction Error: {accuracy:.2f}%")

if __name__ == "__main__":
    main()