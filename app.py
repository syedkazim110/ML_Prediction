from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scalar
model = joblib.load('model.pkl')
scalar = joblib.load('scaler.pkl')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from form data
    crim = float(request.form['crim'])
    zn = float(request.form['zn'])
    indus = float(request.form['indus'])
    chas = int(request.form['chas'])
    nox = float(request.form['nox'])
    rm = float(request.form['rm'])
    age = float(request.form['age'])
    dis = float(request.form['dis'])
    rad = int(request.form['rad'])
    tax = int(request.form['tax'])
    ptratio = float(request.form['ptratio'])
    b = float(request.form['b'])
    lstat = float(request.form['lstat'])

    # Preprocess the features
    features = np.array([crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]).reshape(1, -1)
    scaled_features = scalar.transform(features)

    # Make prediction
    prediction = model.predict(scaled_features)

    # Prepare response
    response = {'prediction': prediction[0]}  # Assuming a single prediction

    return jsonify(response)

# Define a home route for rendering the HTML form
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
