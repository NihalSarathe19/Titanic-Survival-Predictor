
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

# Load the trained model, scaler, and feature names
with open('titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        if request.method == 'POST':
            # For JSON requests
            if request.is_json:
                data = request.get_json()
            else:
                # For form data
                data = request.form.to_dict()

            # Extract features
            age = float(data.get('age', 30))
            fare = float(data.get('fare', 32))
            sibsp = int(data.get('sibsp', 0))
            parch = int(data.get('parch', 0))
            sex_male = 1 if data.get('sex', 'male').lower() == 'male' else 0
            pclass = int(data.get('pclass', 3))
            embarked = data.get('embarked', 'S').upper()
            has_cabin = 1 if data.get('has_cabin', 'no').lower() == 'yes' else 0

            # Create feature vector (matching training features)
            features = np.zeros(len(feature_names))

            # Set basic features (these will be scaled)
            numerical_features = {
                'Age': age,
                'Fare': fare, 
                'SibSp': sibsp,
                'Parch': parch,
                'Family_Size': sibsp + parch + 1
            }

            # Create full feature vector
            feature_dict = {
                'Age': age,
                'SibSp': sibsp,
                'Parch': parch,
                'Fare': fare,
                'Has_Cabin': has_cabin,
                'Sex_Male': sex_male,
                'Embarked_C': 1 if embarked == 'C' else 0,
                'Embarked_Q': 1 if embarked == 'Q' else 0,
                'Embarked_S': 1 if embarked == 'S' else 0,
                'Pclass_1': 1 if pclass == 1 else 0,
                'Pclass_2': 1 if pclass == 2 else 0,
                'Pclass_3': 1 if pclass == 3 else 0,
                'Family_Size': sibsp + parch + 1,
                'Age_Child': 1 if age <= 12 else 0,
                'Age_Teen': 1 if 12 < age <= 18 else 0,
                'Age_Adult': 1 if 18 < age <= 30 else 0,
                'Age_Middle_Age': 1 if 30 < age <= 50 else 0,
                'Age_Senior': 1 if age > 50 else 0,
                'Fare_Low': 1 if fare <= 7.91 else 0,
                'Fare_Medium': 1 if 7.91 < fare <= 14.45 else 0,
                'Fare_High': 1 if 14.45 < fare <= 31 else 0,
                'Fare_Very_High': 1 if fare > 31 else 0,
                'Is_Alone': 1 if (sibsp + parch + 1) == 1 else 0
            }

            # Create input array
            input_features = np.array([[feature_dict[name] for name in feature_names]])

            # Scale numerical features
            numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'Family_Size']
            numerical_indices = [feature_names.index(col) for col in numerical_cols]

            input_scaled = input_features.copy()
            input_scaled[0, numerical_indices] = scaler.transform(
                input_features[:, numerical_indices].reshape(1, -1)
            )[0]

            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]

            # Prepare response
            result = {
                'prediction': int(prediction),
                'survival_probability': float(prediction_proba[1]),
                'death_probability': float(prediction_proba[0]),
                'message': 'Survived' if prediction == 1 else 'Did not survive'
            }

            return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()
        passengers = data.get('passengers', [])

        results = []
        for passenger in passengers:
            # Process each passenger (similar to single prediction)
            # ... (implementation details)
            pass

        return jsonify({'predictions': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
