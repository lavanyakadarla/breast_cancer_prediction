from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

# loading model
model = pickle.load(open(r'C:\Users\katta\Downloads\Breast-Cancer-Detection-using-Machine-Learning-With-App-master\Breast-Cancer-Detection-using-Machine-Learning-With-App-master\models\model.pkl', 'rb'))

# flask app
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['feature1']),
            float(request.form['feature2']),
            float(request.form['feature3']),
            float(request.form['feature4'])
        ]
    except (ValueError, KeyError):
        return render_template('index.html', message=['Please enter valid numbers for all features.'])

    np_features = np.array([features], dtype=np.float32)
    pred = model.predict(np_features)
    message = ['Cancrouse' if pred[0] == 1 else 'Not Cancrouse']
    return render_template('index.html', message=message)





if __name__ == '__main__':
    app.run(debug=True)

