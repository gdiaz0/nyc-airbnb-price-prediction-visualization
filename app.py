from flask import Flask, request, jsonify  # Import Flask to create a web app
import pandas as pd  # Import pandas to read CSV
import joblib  # Import joblib to save and load the model
from sklearn.linear_model import LinearRegression  # Import the machine learning model
from sklearn.model_selection import train_test_split  # Helps split data for training
from sklearn.preprocessing import StandardScaler  # For feature scaling
import numpy as np  # For numerical operations

app = Flask(__name__)  # Create a Flask web app
CSV_FILE = "listings.csv"  # Name of the CSV file
MODEL_FILE = "model.pkl"  # Name of the saved model file
SCALER_FILE = "scaler.pkl"  # Name of the saved scaler file

def preprocess_data(df):
    """
    Preprocess the Airbnb dataset.
    Handles missing values, outliers, and feature transformations.
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_processed = df.copy()
    
    # Clean price column: remove '$' and ',' characters, convert to float
    df_processed['price'] = df_processed['price'].replace({'\\$': '', ',': ''}, regex=True).astype(float)
    
    # Handle missing values
    # For numerical columns, fill with median
    numerical_columns = ['bedrooms', 'bathrooms', 'accommodates']
    for col in numerical_columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # Remove outliers using IQR method
    for col in numerical_columns + ['price']:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_processed = df_processed[(df_processed[col] >= lower_bound) & (df_processed[col] <= upper_bound)]
    
    # Select features for the model
    features = ['bedrooms', 'bathrooms', 'accommodates']
    X = df_processed[features]
    y = df_processed['price']
    
    return X, y

def load_and_train():
    """
    Load data, preprocess it, and train the model.
    """
    try:
        # Load the data
        df = pd.read_csv(CSV_FILE)
        
        # Preprocess the data
        X, y = preprocess_data(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Calculate and print model performance
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        # Save the model and scaler
        joblib.dump(model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        
        return f"Model trained successfully! Training R²: {train_score:.3f}, Test R²: {test_score:.3f}"
    
    except Exception as e:
        return f"Error during training: {str(e)}"

# Endpoint to reload data and train model
@app.route('/reload', methods=['POST'])
def reload():
    message = load_and_train()  # Train model
    return jsonify({"message": message})  # Return a success message in JSON format

# Endpoint to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the model and scaler
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        
        # Get input from user
        data = request.get_json()
        
        # Extract input values or use default values
        bedrooms = float(data.get('bedrooms', 1))
        bathrooms = float(data.get('bathrooms', 1))
        accommodates = float(data.get('accommodates', 1))
        
        # Create feature array and scale it
        X = [[bedrooms, bathrooms, accommodates]]
        X_scaled = scaler.transform(X)
        
        # Make prediction
        price_prediction = model.predict(X_scaled)[0]
        
        return jsonify({
            "predicted_price": float(price_prediction),
            "input_features": {
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "accommodates": accommodates
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Run the Flask app on port 5001 