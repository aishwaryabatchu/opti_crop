# from flask import Flask, render_template, request
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Loaded Predictive Model Binary Ingestion
# with open('models/crop_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         input_data = np.array([[
#             float(request.form['N']), 
#             float(request.form['P']), 
#             float(request.form['K']),
#             float(request.form['temperature']), 
#             float(request.form['humidity']),
#             float(request.form['ph']), 
#             float(request.form['rainfall'])
#         ]])
#         prediction = model.predict(input_data)[0]
#         return render_template('index.html', prediction=prediction.upper())

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('models/crop_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load label encoder
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

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

        # Predict numeric label
        prediction = model.predict(input_data)[0]

        # Convert back to crop name
        crop_name = le.inverse_transform([prediction])[0]

        return render_template('index.html', prediction=crop_name.upper())

# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)