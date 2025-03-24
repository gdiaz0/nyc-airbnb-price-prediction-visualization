# Airbnb Price Prediction Model

This project implements a machine learning model to predict Airbnb listing prices based on various features. The model uses Linear Regression and includes comprehensive data preprocessing steps.

## Project Overview

This project was developed as part of a machine learning assignment, focusing on:
1. Data preprocessing and cleaning
2. Model training and evaluation
3. API development for model deployment

## Features

- Data preprocessing with handling of missing values and outliers
- Feature scaling using StandardScaler
- Linear Regression model for price prediction
- RESTful API endpoints for model training and prediction
- Error handling and input validation

## Data Preprocessing Steps

1. Price cleaning: Removes '$' and ',' characters, converts to float
2. Missing value handling: Fills missing values with median for numerical columns
3. Outlier removal: Uses IQR method to remove outliers from all numerical columns
4. Feature scaling: Standardizes features using StandardScaler

## Model Details

The project uses a Linear Regression model with the following features:
- Number of bedrooms
- Number of bathrooms
- Number of people it accommodates

The model includes:
- Feature scaling for better performance
- Train/test split (80/20)
- Performance evaluation using R² scores
- Model persistence using joblib

## API Endpoints

### 1. Train Model
```
POST /reload
```
Trains the model using the current dataset. Returns training and test R² scores.

### 2. Make Predictions
```
POST /predict
```
Makes price predictions based on input features.

Example request body:
```json
{
    "bedrooms": 2,
    "bathrooms": 1,
    "accommodates": 4
}
```

## Setup and Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Flask application:
```bash
python app.py
```

The application will be available at `http://127.0.0.1:5001`

## Dependencies

- Flask==3.0.2
- pandas
- scikit-learn
- joblib
- numpy
- Flask-SQLAlchemy==3.1.1
- SQLAlchemy==2.0.35
- requests==2.26.0
- flasgger==0.9.7.1
- gunicorn==20.1.0
- pytest==8.3.3
- pytest-flask==1.3.0

## Model Performance

The model's performance is evaluated using R² scores on both training and test sets. These scores are displayed when the model is trained using the `/reload` endpoint.

## Project Structure

```
.
├── app.py              # Main Flask application
├── requirements.txt    # Project dependencies
├── README.md          # Project documentation
├── .gitignore         # Git ignore rules
└── listings.csv       # Dataset (not included in repository)
```

## Testing

The project includes pytest configuration for testing the API endpoints. Run tests using:
```bash
pytest
``` 