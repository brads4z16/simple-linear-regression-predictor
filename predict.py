import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

class TaxiFarePredictor:
    def __init__(self, model_path='trained_model.pkl'):
        """
        Initialize the predictor with a saved model
        
        Args:
            model_path (str): Path to the saved model file
        """
        self.model = None
        self.label_encoders = {}
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load the trained model and label encoders
        
        Args:
            model_path (str): Path to the saved model file
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.label_encoders = model_data['label_encoders']
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict_fare(self, trip_miles, trip_seconds, company, tip_rate):
        """
        Predict fare for a new trip
        
        Args:
            trip_miles (float): Distance of the trip in miles
            trip_seconds (int): Duration of the trip in seconds
            company (str): Taxi company name
            tip_rate (float): Expected tip rate
            
        Returns:
            float: Predicted fare
        """
        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'TRIP_MILES': [trip_miles],
            'TRIP_SECONDS': [trip_seconds],
            'COMPANY': [company],
            'TIP_RATE': [tip_rate]
        })
        
        # Encode categorical variables using the stored encoders
        for feature, encoder in self.label_encoders.items():
            input_data[feature] = encoder.transform(input_data[feature])
        
        # Make prediction
        predicted_fare = self.model.predict(input_data)[0]
        return round(predicted_fare, 2)

def main():
    # Initialize predictor
    predictor = TaxiFarePredictor()
    
    # Example predictions
    trips = [
        {
            'description': '5-mile, 15-minute trip with Sun Taxi',
            'trip_miles': 5.0,
            'trip_seconds': 900,
            'company': 'Sun Taxi',
            'tip_rate': 0.15
        },
        {
            'description': '10-mile, 30-minute trip with Flash Cab',
            'trip_miles': 10.0,
            'trip_seconds': 1800,
            'company': 'Flash Cab',
            'tip_rate': 0.20
        }
    ]
    
    # Make predictions for each trip
    for trip in trips:
        predicted_fare = predictor.predict_fare(
            trip_miles=trip['trip_miles'],
            trip_seconds=trip['trip_seconds'],
            company=trip['company'],
            tip_rate=trip['tip_rate']
        )
        print(f"\nPredicted fare for {trip['description']}: ${predicted_fare}")

if __name__ == "__main__":
    main() 