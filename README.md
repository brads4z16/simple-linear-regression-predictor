# Chicago Taxi Fare Prediction Model

This project implements a machine learning model to predict taxi fares in Chicago using linear regression. The model is inspired by and uses data from the [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/linear-regression/programming-exercise). Although it is inspired by, I have implemented it differently from the crash course, by using standard libraries which achieve better results. 

## Model Overview

The model predicts taxi fares based on several features:
- Trip distance (miles)
- Trip duration (seconds)
- Taxi company
- Payment type
- Tip rate

### Performance Metrics
- R² Score: 0.959 (95.9% of variance explained)
- RMSE: $3.44 (average prediction error)
- MSE: 11.89 (mean squared error)

## Features

- **Data Processing**: Handles both numerical and categorical features
- **Model Training**: Uses scikit-learn's LinearRegression
- **Model Persistence**: Saves and loads trained models
- **Prediction Interface**: Easy-to-use interface for making predictions
- **Performance Visualization**: Includes plotting capabilities

## Example Prediction

```python
predicted_fare = model.predict_fare(
    trip_miles=5.0,
    trip_seconds=900,  # 15 minutes
    company='Sun Taxi',
    payment_type='Credit Card',
    tip_rate=0.15
)
```

## Model Architecture

The model uses a linear regression approach:
```
FARE = β₀ + β₁(TRIP_MILES) + β₂(TRIP_SECONDS) + β₃(COMPANY) + β₄(PAYMENT_TYPE) + β₅(TIP_RATE)
```

Where:
- β₀ is the intercept
- β₁ through β₅ are the coefficients for each feature
- Categorical variables (COMPANY, PAYMENT_TYPE) are encoded using LabelEncoder

## Data Source

The training data comes from the Google Machine Learning Crash Course's linear regression programming exercise, which provides a dataset of Chicago taxi trips. This dataset includes various features about taxi trips and their corresponding fares.

## Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0 