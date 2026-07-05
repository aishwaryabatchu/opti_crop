from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Loaded Predictive Model Binary Ingestion
with open('models/crop_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = np.array([[
            float(request.form['N']), 
            float(request.form['P']), 
            float(request.form['K']),
            float(request.form['temperature']), 
            float(request.form['humidity']),
            float(request.form['ph']), 
            float(request.form['rainfall'])
        ]])
        prediction = model.predict(input_data)[0]
        return render_template('index.html', prediction=prediction.upper())

if __name__ == '__main__':
    app.run(debug=True)