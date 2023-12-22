from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and encoder
model = joblib.load('/home/Kanex/Documents/dr An/ML(SVM)/trained_model4.pkl')
encoder = joblib.load('/home/Kanex/Documents/dr An/ML(SVM)/trained_model4.pkl')

label_encoder = LabelEncoder()

def preprocess_input(diagnosis, drugs):

    user_data = pd.DataFrame({'Diagnosis': [diagnosis], 'Drugs': [drugs]})
    
    diagnosis_encoded =label_encoder.fit_transform(user_data['Diagnosis'])
    drugs_encoded =label_encoder.fit_transform(user_data['Drugs'])

    # Concatenate encoded features
    user_data_encoded = pd.DataFrame({'Diagnosis': diagnosis_encoded, 'Drugs': drugs_encoded})
   
    
    return user_data_encoded

# Streamlit UI
def main():
    st.title("Machine Learning Model with Streamlit")

    # User input
    diagnosis = st.text_input("Enter Diagnosis:")
    drugs = st.text_input("Enter Drugs:")

    if st.button("Predict"):
        user_data_encoded = preprocess_input(diagnosis, drugs)

        prediction = model.predict(user_data_encoded)

        st.success(f"Prediction: {prediction[0]}")

if __name__ == "__main__":
   main()

