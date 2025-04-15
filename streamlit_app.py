import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import gzip
import io

st.title("NYC Airbnb Price Data Visualizer")

#  Cache data to avoid reloading every time the app is ran
@st.cache_data
def load_data():
    try:
        # Trying to load from local file first, if it is not there then an error message will be shown that says "File not found"
        # The next step if not found would be to download the data from the website
        df = pd.read_csv("listings.csv")
        # Clean price column if it exists
        if 'price' in df.columns:
            # Remove '$' and ',' from price, convert to float
            df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
        return df
    except FileNotFoundError:
        st.warning("Local file not found. Downloading data from Inside Airbnb...")
        url = "http://data.insideairbnb.com/united-states/ny/new-york-city/2024-01-08/data/listings.csv.gz"
        try:
            response = requests.get(url)
            response.raise_for_status()
            # Decompress gz and make it into a pandas df
            with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
                df = pd.read_csv(f)
            # Clean price column
            if 'price' in df.columns:
                df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
            # Save the data locally for future use
            df.to_csv("listings.csv", index=False)
            return df
        except Exception as e:
            st.error(f"Error downloading data: {str(e)}")
            return pd.DataFrame()

# Loading the data
listings = load_data()

if listings.empty:
    st.error("No data available. Please check your internet connection or ensure the data file exists.")
else:
    st.sidebar.header("Filters")
    
    # Price range filter
    min_price = st.sidebar.number_input("Minimum Price", min_value=0, value=0)
    max_price = st.sidebar.number_input("Maximum Price", min_value=0, value=1000)
    
    # Room type filter
    room_types = listings['room_type'].unique()
    selected_room_types = st.sidebar.multiselect("Room Types", room_types, default=room_types)
    
    # Applying the filters
    filtered_data = listings[
        (listings['price'] >= min_price) & 
        (listings['price'] <= max_price) & 
        (listings['room_type'].isin(selected_room_types))
    ]
    
    # Main content
    st.header("Data Overview")
    
    # Including basic statistics through streamlit 
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Listings", len(filtered_data))
    with col2:
        avg_price = filtered_data['price'].mean()
        st.metric("Average Price", f"${avg_price:.2f}")
    with col3:
        st.metric("Unique Neighborhoods", filtered_data['neighbourhood'].nunique())
    
    # Visualization type for users to choose from
    viz_type = st.selectbox(
        "Choose Visualization Type",
        ["Price Distribution", "Room Type Distribution", "Neighborhood Analysis", "Price vs. Reviews"]
    )
    
    if viz_type == "Price Distribution":
        st.subheader("Price Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=filtered_data, x='price', bins=50)
        plt.xlim(0, 1000)  
        st.pyplot(fig)
        
    elif viz_type == "Room Type Distribution":
        st.subheader("Room Type Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=filtered_data, x='room_type')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    elif viz_type == "Neighborhood Analysis":
        st.subheader("Neighborhood Analysis")
        top_n = st.slider("Number of Top Neighborhoods", 5, 20, 10)
        neighborhood_stats = filtered_data.groupby('neighbourhood').agg({
            'price': 'mean',
            'id': 'count'
        }).rename(columns={'id': 'count'}).sort_values('count', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=neighborhood_stats.reset_index(), x='neighbourhood', y='count')
        plt.xticks(rotation=90)
        st.pyplot(fig)
        
        # Show the average prices for top neighborhoods
        st.write("Average Prices by Neighborhood:")
        st.dataframe(neighborhood_stats)
        
    elif viz_type == "Price vs. Reviews":
        st.subheader("Price vs. Number of Reviews")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=filtered_data, x='number_of_reviews', y='price')
        plt.xlim(0, 200)  
        plt.ylim(0, 1000)  
        st.pyplot(fig)
    
    # Raw data check box that allows users to see the raw data if they choose to
    if st.checkbox("Show Raw Data"):
        st.subheader("Raw Data")
        st.dataframe(filtered_data)
