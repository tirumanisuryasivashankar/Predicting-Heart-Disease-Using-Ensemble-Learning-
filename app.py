import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import numpy as np

# --- Constants ---
# Use the local CSV file provided by the user
DATA_PATH = 'heart_disease_uci.csv'
# Filename for the saved machine learning model
MODEL_PATH = 'heart_disease_model.pkl'
# Filename to save the list of training columns
COLUMNS_PATH = 'model_columns.pkl'

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data():
    """
    Loads the local heart disease dataset, cleans and preprocesses it,
    and prepares the data for model training. This version is for MULTICLASS classification.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Error: The data file '{DATA_PATH}' was not found. "
            "Please make sure it's in the same directory as app.py."
        )
    
    # Load the dataset from the local CSV file, treating '?' as missing values
    df = pd.read_csv(DATA_PATH, na_values='?')
    
    # --- Data Cleaning and Formatting ---
    # Drop columns that are not useful for prediction
    df = df.drop(columns=['id', 'dataset'])
    
    # Rename the target column from 'num' to 'target' for clarity
    df = df.rename(columns={'num': 'target'})
    
    # Convert binary categorical text columns to numerical format
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    df['fbs'] = df['fbs'].astype(bool).astype(int) # Convert TRUE/FALSE to 1/0
    df['exang'] = df['exang'].astype(bool).astype(int) # Convert TRUE/FALSE to 1/0
    
    # --- Handle missing values based on data type ---
    # Separate columns by data type
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Impute missing values for numeric columns with the median
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Impute missing values for categorical columns with the mode
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # --- KEY CHANGE FOR MULTICLASS CLASSIFICATION ---
    # The 'target' column has values from 0 to 4.
    # We will KEEP these values to predict the specific stage of the disease.
    # The line that converted this to a binary problem has been removed.
    
    # Separate features (X) and the target variable (y)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # One-Hot Encode categorical features to convert them into a numerical format
    X = pd.get_dummies(X, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)
    
    return X, y

# --- Model Training and Saving ---
def train_and_evaluate_model(X, y):
    """
    Splits the data, trains a RandomForestClassifier for multiclass prediction, 
    evaluates it, and saves the trained model and its columns to disk.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize the Random Forest Classifier (it handles multiclass problems automatically)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model on the training data
    print("Training the multiclass model...")
    model.fit(X_train, y_train)
    
    # --- Model Evaluation ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n--- Model Evaluation (Multiclass) ---")
    print(f"Model Accuracy: {accuracy:.2f} (Predicting the exact stage is harder, so lower accuracy is expected)")
    
    # Define labels for the classification report
    target_names = [
        'No Disease (0)', 
        'Stage 1 Disease (1)', 
        'Stage 2 Disease (2)', 
        'Stage 3 Disease (3)', 
        'Stage 4 Disease (4)'
    ]
    
    print("\nClassification Report:")
    # Use the actual labels present in the test set to avoid errors if a class is missing
    unique_labels = sorted(y_test.unique())
    report_labels = [target_names[i] for i in unique_labels]
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=report_labels, zero_division=0))
    print("---------------------------------------\n")
    
    # --- Save the Model and Columns ---
    joblib.dump(model, MODEL_PATH)
    print(f"Multiclass model saved to '{MODEL_PATH}'")
    joblib.dump(X.columns, COLUMNS_PATH)
    print(f"Model columns saved to '{COLUMNS_PATH}'")


# --- Prediction Function ---
def predict(input_data):
    """
    Loads the saved model and makes a multiclass prediction on new, unseen data.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
        raise FileNotFoundError("Model or columns file not found. Please train the model first by running `python app.py`.")
        
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    
    # Create a DataFrame from the input data (a dictionary)
    input_df = pd.DataFrame([input_data])
    
    # --- Preprocess input data to match training format ---
    input_df['sex'] = input_df['sex'].map({'Male': 1, 'Female': 0})
    input_df['fbs'] = input_df['fbs'].astype(bool).astype(int)
    input_df['exang'] = input_df['exang'].astype(bool).astype(int)
    
    input_df = pd.get_dummies(input_df, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    # Make a prediction (will return a value from 0 to 4)
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    return prediction[0], prediction_proba[0]


# --- Main Execution Block ---
if __name__ == "__main__":
    # Check if the model already exists. If not, train it.
    if not os.path.exists(MODEL_PATH):
        print("Model file not found. Starting training process for multiclass model...")
        try:
            X, y = load_and_preprocess_data()
            train_and_evaluate_model(X, y)
        except FileNotFoundError as e:
            print(e)
            exit()
    else:
        print(f"Multiclass model '{MODEL_PATH}' found. Ready for predictions.")
        
    # --- Example Usage of the Prediction Function ---
    print("\n--- Making a Multiclass Prediction for a Sample Patient ---")
    
    sample_patient_data = {
        'age': 52, 'sex': 'Male', 'cp': 'non-anginal', 'trestbps': 125, 'chol': 212,
        'fbs': False, 'restecg': 'normal', 'thalch': 168, 'exang': False, 'oldpeak': 1.0,
        'slope': 'upsloping', 'ca': 2, 'thal': 'reversable defect'
    }

    try:
        prediction_result, prediction_probabilities = predict(sample_patient_data)
        
        # Map the numeric prediction to a meaningful string
        disease_map = {
            0: 'No Heart Disease',
            1: 'Stage 1 Heart Disease',
            2: 'Stage 2 Heart Disease',
            3: 'Stage 3 Heart Disease',
            4: 'Stage 4 Heart Disease'
        }
        
        result_text = disease_map.get(prediction_result, "Unknown Prediction")
            
        print(f"Prediction: {result_text}")
        print(f"Confidence across stages: {np.round(prediction_probabilities*100, 2)}%")
    
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

