import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pydeck

joblib.load("US_House_Cleaning_Pipeline.pkl")
joblib.load("US_House_price_model.pkl")

st.title("US House price predictor")
st.write(
    "Want to know the price of your dream house? Try out our AI to help give you a good estimate!"
)

OverallQual = st.number_input('Overall Quality', 1, 10, 1)
GrLivArea = st.number_input('Garage Living Area', 0, 6000)
TotalBsmtSF = st.number_input('Total Basement Area', 0, 7000)
GarageArea = st.number_input('Garage Area', 0, 2000)
GarageCars = st.number_input('Garage Cars', 0, 4, 1)

if st.button("Predict Detailed Price"):
    features = np.array([[
                OverallQual, GrLivArea, TotalBsmtSF, GarageArea, GarageCars
    ]])
    price = US_House_price_model.predict(features)
    price = np.exp(price)

    st.success(f"Estimated Price: ${price}")
