import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --- Constants ---
# Filename for the saved machine learning model
MODEL_PATH = 'heart_disease_model.pkl'
# Filename to save the list of training columns
COLUMNS_PATH = 'model_columns.pkl'

# --- Prediction Function (Adapted for Multiclass) ---
def predict(input_data):
    """
    Loads the saved multiclass model and makes a prediction on new, unseen data.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
        st.error(
            "Model files not found. "
            "Please run `python app.py` in your terminal first to train the multiclass model."
        )
        return None, None

    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    
    input_df = pd.DataFrame([input_data])
    
    # --- Preprocess input data to match training format ---
    input_df['sex'] = input_df['sex'].map({'Male': 1, 'Female': 0})
    input_df['fbs'] = input_df['fbs'].map({'> 120 mg/dl': True, '<= 120 mg/dl': False}).astype(bool).astype(int)
    input_df['exang'] = input_df['exang'].map({'Yes': True, 'No': False}).astype(bool).astype(int)
    
    input_df = pd.get_dummies(input_df, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    # Make a prediction (will return a value from 0 to 4)
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    return prediction[0], prediction_proba[0]

# --- Streamlit User Interface ---
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title('🩺 Multiclass Heart Disease Prediction App')
st.markdown("Enter patient details to predict the specific stage of heart disease.")

# --- Define the mapping for prediction results ---
disease_map = {
    0: 'No Heart Disease',
    1: 'Stage 1 Heart Disease',
    2: 'Stage 2 Heart Disease',
    3: 'Stage 3 Heart Disease',
    4: 'Stage 4 Heart Disease'
}

# Create two columns for a better layout
col1, col2 = st.columns(2)

with col1:
    st.header("Patient Demographics")
    age = st.number_input('Age', min_value=1, max_value=120, value=52)
    sex = st.selectbox('Sex', ('Male', 'Female'))

    st.header("Medical History")
    cp = st.selectbox('Chest Pain Type (cp)', ('typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'))
    trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=50, max_value=250, value=125)
    chol = st.number_input('Serum Cholesterol in mg/dl (chol)', min_value=100, max_value=600, value=212)
    fbs = st.selectbox('Fasting Blood Sugar (fbs)', ('> 120 mg/dl', '<= 120 mg/dl'))
    restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', ('normal', 'st-t abnormality', 'lv hypertrophy'))

with col2:
    st.header("Exercise & Test Results")
    thalch = st.number_input('Maximum Heart Rate Achieved (thalch)', min_value=50, max_value=250, value=168)
    exang = st.selectbox('Exercise Induced Angina (exang)', ('No', 'Yes'))
    oldpeak = st.number_input('ST depression (oldpeak)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox('Slope of ST segment (slope)', ('upsloping', 'flat', 'downsloping'))
    ca = st.number_input('Number of major vessels colored by flourosopy (ca)', min_value=0, max_value=4, value=2)
    thal = st.selectbox('Thalassemia (thal)', ('normal', 'fixed defect', 'reversable defect'))

# --- Prediction Button and Results ---
st.write("")
if st.button('Predict Disease Stage', key='predict_button'):
    patient_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs == '> 120 mg/dl', 'restecg': restecg, 'thalch': thalch,
        'exang': exang == 'Yes', 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }

    prediction, probabilities = predict(patient_data)

    if prediction is not None:
        st.write("---")
        st.header("Prediction Result")
        
        result_text = disease_map.get(prediction, "Unknown Prediction")
        
        # Display the result with appropriate color
        if prediction > 0:
            st.error(f'**Result: {result_text}**')
        else:
            st.success(f'**Result: {result_text}**')
        
        st.subheader("Prediction Confidence Breakdown")
        # Create a DataFrame for better visualization of probabilities
        prob_df = pd.DataFrame({
            'Disease Stage': [disease_map[i] for i in range(len(probabilities))],
            'Confidence': [f"{p*100:.2f}%" for p in probabilities]
        })
        st.table(prob_df)

