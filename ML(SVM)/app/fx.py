# import yfinance as yf
# import streamlit as st 
# import pandas as pd

# st.write("""
# #simple fx price app""")

# tickerSymbol = "eurusd"

# tickerData = yf.Ticker(tickerSymbol)
# tickerDf = tickerData.history(period="1d" , start="2017-10-31" , end ="2023-10-15")

# st.line_chart(tickerDf.Close)
# st.line_chart(tickerDf.Volume)





import streamlit as st
import pandas as pd
from sklearn.externals import joblib

# Load the trained model and encoder
model, encoder = joblib.load('antibiotic_model.pkl')

# Function to preprocess user input
def preprocess_input(diagnosis, drugs):
    diagnosis_encoded = encoder.transform([diagnosis])[0]
    drugs_encoded = encoder.transform([drugs])[0]
    return [diagnosis_encoded, drugs_encoded]

# Streamlit UI
def main():
    st.title("Antibiotic Combination Effectiveness Prediction")

    # User input
    diagnosis = st.text_input("Enter Diagnosis:")
    drugs = st.text_input("Enter Drugs:")

    if st.button("Predict"):
        try:
            # Preprocess user input
            user_data_encoded = preprocess_input(diagnosis, drugs)

            # Make prediction
            prediction = model.predict([user_data_encoded])[0]

            # Display result
            result = "Effective" if prediction == 'a' else "Ineffective"
            st.success(f"Prediction: {result}")
        except Exception as e:
            # Handle prediction error
            st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # mine
    
    
    
    import streamlit as st
import pandas as pd
import joblib

# Load the trained model and encoder
model = joblib.load('/home/Kanex/Documents/dr An/ML(SVM)/trained_model4.pkl')

# Function to preprocess user input
# def preprocess_input(diagnosis, drugs):
#     diagnosis_encoded = encoder.transform([diagnosis])
#     drugs_encoded = encoder.transform([drugs])
#     return [diagnosis_encoded, drugs_encoded]

# Streamlit UI
def main():
    st.title("Antibiotic Combination Effectiveness Prediction")

    # User input
    diagnosis = st.text_input("Enter Diagnosis:")
    drugs = st.text_input("Enter Drugs:")

    if st.button("Predict"):
        try:
            # Preprocess user input
            # user_data_encoded = preprocess_input(diagnosis, drugs)

            # Make prediction
            prediction = model.predict([diagnosis, drugs])

            # Display result
            result = "Effective" if prediction == '' else "Ineffective"
            st.success(f"Prediction: {result}")
        except Exception as e:
            # Handle prediction error
            st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()

