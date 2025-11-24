from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load all saved models
models = {
    'catboost': pickle.load(open('catboost_model.pkl', 'rb')),
    # 'random_forest': pickle.load(open('random_forest_model.pkl', 'rb')),
    'knn_keighbors': pickle.load(open('knn_keighbors_model.pkl', 'rb')),
    'gbr_gradient': pickle.load(open('gbr_gradient_model.pkl', 'rb')),
    # 'xgb_regressor': pickle.load(open('xgb_regressor_model.pkl', 'rb')),
    'linear_regression': pickle.load(open('linear_regression_model.pkl', 'rb')),
    # 'lgb_regressor': pickle.load(open('lgb_regressor_model.pkl', 'rb')),
    # 'neural_network': pickle.load(open('neural_model.pkl', 'rb')),
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_name = data['model']  # Get the desired model name from the request

    if model_name in models:
        model = models[model_name]
        input_df = pd.DataFrame([data['features']])  # Corrected line
        prediction = model.predict(input_df)[0]
        return jsonify({'prediction': prediction})
    else:
        return jsonify({'error': 'Invalid model name'}), 400  # Return error if model not found

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)