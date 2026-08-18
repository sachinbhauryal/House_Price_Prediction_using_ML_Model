import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# -----------------------------------
# Page Title
# -----------------------------------

st.title("🏠 House Price Prediction using Machine Learning")

st.image(
    "https://images.pexels.com/photos/33305255/pexels-photo-33305255.jpeg"
)

st.write("""
House Price Prediction is a supervised machine learning regression problem
in which a model predicts the price of a house based on different features.
""")


# -----------------------------------
# Load Dataset
# -----------------------------------

data = fetch_california_housing()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="Price")


# -----------------------------------
# Train-Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------------
# Feature Scaling
# -----------------------------------

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------------
# Train Model
# -----------------------------------

model = LinearRegression()

model.fit(X_train_scaled, y_train)


# -----------------------------------
# Model Evaluation
# -----------------------------------

y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

st.subheader("Model Performance")

st.write("R² Score:", round(r2, 4))
st.write("Mean Squared Error:", round(mse, 4))


# -----------------------------------
# User Input
# -----------------------------------

# -----------------------------------
# User Input using Sliders
# -----------------------------------

# -----------------------------------
# Sidebar Input
# -----------------------------------

st.sidebar.title("🏠 House Details")
st.sidebar.image('https://i.pinimg.com/originals/b1/cf/53/b1cf530756b23930aab499c1a11b443c.gif')
st.sidebar.write("Enter the house information below:")

MedInc = st.sidebar.slider(
    "Median Income",
    0.0, 20.0, 5.0, 0.1
)

HouseAge = st.sidebar.slider(
    "House Age",
    1.0, 52.0, 20.0, 1.0
)

AveRooms = st.sidebar.slider(
    "Average Rooms",
    1.0, 15.0, 5.0, 0.1
)

AveBedrms = st.sidebar.slider(
    "Average Bedrooms",
    0.5, 5.0, 1.0, 0.1
)

Population = st.sidebar.slider(
    "Population",
    1.0, 40000.0, 1000.0, 100.0
)

AveOccup = st.sidebar.slider(
    "Average Occupancy",
    1.0, 20.0, 3.0, 0.1
)

Latitude = st.sidebar.slider(
    "Latitude",
    32.0, 42.0, 35.0, 0.1
)

Longitude = st.sidebar.slider(
    "Longitude",
    -125.0, -114.0, -120.0, 0.1
)


# -----------------------------------
# Prediction
# -----------------------------------

if st.button("🔮 Predict House Price"):

    all_value = []

    all_value.append(MedInc)
    all_value.append(HouseAge)
    all_value.append(AveRooms)
    all_value.append(AveBedrms)
    all_value.append(Population)
    all_value.append(AveOccup)
    all_value.append(Latitude)
    all_value.append(Longitude)

    input_data = pd.DataFrame(
        [all_value],
        columns=data.feature_names
    )

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.success(
        f"🏠 Predicted House Price: ${prediction[0] * 100000:,.2f}"
    )