import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from PIL import Image

# Load the dataset
df = pd.read_csv('final_cleaned.csv')
df.drop(columns=["Unnamed: 0"], inplace=True)

# Add custom CSS styles for buttons and select boxes
st.markdown("""
    <style>
    .stSelectbox, .stNumberInput {
        background-color: #f0f0f0;
        color: #333333;
        font-size: 16px;
    }
    .stButton button {
        background-color: #FF4B4B;
        color: white;
        font-size: 20px;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #FF6F6F;
    }
    </style>
""", unsafe_allow_html=True)

# Logo and banner
st.sidebar.image("carDekho-newLogo.svg", use_column_width=True)
banner = Image.open('banner.jpeg')
banner = banner.resize((2500, 900))
st.image(banner)

# Add a description of the Cardheko-like website
st.sidebar.header("About Cardheko")
st.sidebar.write("""
Cardheko is a comprehensive platform that helps users explore, compare, and purchase cars.
With an extensive range of new and used cars, the website offers in-depth reviews, 
pricing details, and expert advice to guide car buyers in making informed decisions. 
Use our price prediction tool to estimate car values based on various features and conditions.
""")

# Streamlit app interface
st.title("Car Price Prediction")

st.header("Enter Car Details")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    # Input fields for the user
    city = st.selectbox('City', df['city'].unique(), key='city')
    fuel_type = st.selectbox('Fuel Type', df['fuel_type'].unique(), key='fuel_type')
    body_type = st.selectbox('Body Type', df['body_type'].unique(), key='body_type')
    km_driven = st.number_input('Kilometers Driven', min_value=0, max_value=500000, step=1000, key='km_driven')
    transmission = st.selectbox('Transmission', df['transmission'].unique(), key='transmission')

with col2:
    owner_no = st.number_input('Owner Number', min_value=1, max_value=5, key='owner_no')
    car_model = st.selectbox('Car Model', df['model'].unique(), key='car_model')
    model_year = st.number_input('Model Year', min_value=int(df['modelyear'].min()), max_value=int(df['modelyear'].max()), key='model_year')
    color = st.selectbox('Color', df['color'].unique(), key='color')

# Fit and store label encoders for categorical columns
categorical_cols = df.select_dtypes(include=[object]).columns
encoders = {}
for column in categorical_cols:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# Split the data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Train the model
model = GradientBoostingRegressor(n_estimators=300, random_state=42, learning_rate=0.2, max_depth=5, min_samples_leaf=4, min_samples_split=2)
model.fit(X_train, y_train)

# Create a dictionary for the input data
input_data = {
    'city': city,
    'fuel_type': fuel_type,
    'body_type': body_type,
    'km_driven': km_driven,
    'transmission': transmission,
    'ownerno': owner_no,
    'model': car_model,
    'modelyear': model_year,
    'color': color
}

# Convert the input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Apply label encoding to the input data using the fitted encoders
for column in input_df.columns:
    if column in encoders:
        le = encoders[column]
        input_df[column] = le.transform(input_df[column])

# Make prediction using the trained model
if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.write(f"Predicted Price: ₹{int(prediction[0]):,}")
