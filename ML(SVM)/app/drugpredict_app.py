import streamlit as st
import pandas as pd
import joblib


loaded_objects = joblib.load('/home/Kanex/Documents/dr An/ML(SVM)/trained_model1.pkl')
model = loaded_objects


# Function to preprocess user input
def preprocess_input(diagnosis, drugs):

    user_data = pd.DataFrame({'Diagnosis': [diagnosis], 'Drugs': [drugs]})

    diagnosis_encoded = pd.get_dummies(user_data['Diagnosis'], columns=diagnosis, drop_first=True)
    drugs_encoded = pd.get_dummies(user_data['Drugs'], columns=drugs, drop_first=True)
    user_data_encoded = pd.concat([diagnosis_encoded, drugs_encoded])

    return user_data_encoded

 
def main():
    st.title("Antibiotic Combination Effectiveness Prediction")

    # User input
    diagnosis = st.text_input("Enter Diagnosis:")
    drugs = st.text_input("Enter Drugs:")

    if st.button("Predict"):
        try:
            user_data_encoded = preprocess_input(diagnosis, drugs)

            prediction = model.predict([user_data]) 

            result = "Effective" if prediction == 'A' else "Ineffective"
            st.success(f"Prediction: {result}")
        except Exception as e:
           
            st.error(f"Error during prediction: {e}")
       
             
if __name__ == "__main__":
    main()
