import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load earthquake data
earthquake_data = pd.read_csv("usgs_main.csv")

# Select features and target
features = ['latitude', 'longitude', 'depth']
target = 'mag'

# Drop rows with missing values
earthquake_data = earthquake_data.dropna()

X = earthquake_data[features]
y = earthquake_data[target]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Define Streamlit app
def main():
    st.title("Earthquake Magnitude Predictor")
    st.warning("Keep feeding data don't get distracted by dimming of screen")
    st.write("Enter earthquake details:")
    latitude = st.number_input("Latitude:")
    longitude = st.number_input("Longitude:")
    depth = st.number_input("Depth:")

    if st.button("Predict"):
        prediction = model.predict([[latitude, longitude, depth]])[0]
        st.success(f"Predicted magnitude: {prediction}")

if __name__ == "__main__":
    main()
