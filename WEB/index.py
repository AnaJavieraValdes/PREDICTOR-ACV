from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model 
import joblib

app = Flask(__name__)

# Cargar el modelo y MinMaxScaler
model = load_model('my_model.h5')
min_max_scaler = joblib.load('min_max_scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [
        int(request.form['gender']),
        float(request.form['age']),
        int(request.form['hypertension']),
        int(request.form['heart_disease']),
        int(request.form['ever_married']),
        int(request.form['work_type']),
        int(request.form['residence_type']),
        float(request.form['avg_glucose_level']),
        float(request.form['bmi']),
        int(request.form['smoking_status'])
    ]
    
    inputs = np.array(data).reshape(1, -1)
    inputs = min_max_scaler.transform(inputs)
    
    predictions = model.predict(inputs)
    
    result = " {:.2f}%".format(predictions[0][0] * 100)
    return render_template('result.html', result=result)


if __name__ == '__main__':
    import os
    os.environ['FLASK_ENV'] = 'development'
    app.run(debug=True)
