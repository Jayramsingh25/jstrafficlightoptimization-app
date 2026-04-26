# ==============================
# BACKEND CODE (app.py)
# ==============================

from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model files
model = joblib.load('svr_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')


# Home Page
@app.route('/')
def home():
    return render_template('index.html')


# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():

    # Get values from frontend
    traffic_density = float(request.form['traffic_density'])
    vehicle_count = float(request.form['vehicle_count'])
    average_speed = float(request.form['average_speed'])
    lane_count = float(request.form['lane_count'])
    peak_hour = float(request.form['peak_hour'])

    # Convert into array
    data = np.array([[
        traffic_density,
        vehicle_count,
        average_speed,
        lane_count,
        peak_hour
    ]])

    # Standardize
    data_scaled = scaler.transform(data)

    # PCA
    data_pca = pca.transform(data_scaled)

    # Predict
    prediction = model.predict(data_pca)

    output = round(prediction[0], 2)

    # Send result to frontend
    return render_template(
        'index.html',
        prediction_text=f'Predicted Green Signal Time: {output} seconds'
    )


# Run App
if __name__ == '__main__':
    app.run(debug=True)