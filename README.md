# NYC Airbnb Price Prediction and Analysis

The initial phase of this project was to implements a machine learning model to predict Airbnb listing prices based on various features. The model uses Linear Regression and includes comprehensive data preprocessing steps. Now, we are building off the previous model to allow for visualization. Visualizing the data can give important insight to the trends of the data and overall increase interpretability. 

## Project Overview

This project was developed as part of a machine learning assignment, focusing on:
1. Data preprocessing and cleaning
2. Model training and evaluation
3. API development for model deployment
4. Creation of streamlit app to visual the data 

## Features
- Creation of an interactive visualization model for NYC Airbnb data
    - This includes metrics on average nights spent, bathrooms, number of reviews
- Users are able to filter through the data based on variables they find most important to them, for example by price range and room type
- Multiple visualization options:
  - Price Distribution
  - Room Type Distribution
  - Neighborhood Analysis
  - Price vs. Reviews Analysis
- Raw data exploration


## Data Preprocessing Steps

1. Clone this repository:
```bash
git clone [your-repository-url]
cd nyc-airbnb-price-predicition
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install the required packages and discrepancies:
```bash
pip install -r requirements.txt
```

## Streamlit App 

Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

The app will then be opned on a web browser at `http://localhost:8501`.

## Data Collection Source

The data is sourced from Inside Airbnb (http://insideairbnb.com/), specifically the NYC listings dataset from January 2024.

## Project Structure

- `streamlit_app.py`: Main application file
- `requirements.txt`: Python package dependencies
- `README.md`: Project documentation

## Deployment

This app can be deployed on Streamlit Cloud. To deploy:
1. Push your code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect your GitHub repository
4. Deploy the app

## License

This project is licensed under the MIT License - see the LICENSE file for details. 