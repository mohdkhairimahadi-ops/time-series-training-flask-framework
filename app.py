# app.py
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

model = load_model('model/model.h5', compile=False)
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html.jinja')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        temp_str = request.form.get('temp_values', '').strip()
        hum_str  = request.form.get('hum_values', '').strip()

        if not temp_str or not hum_str:
            return render_template('index.html.jinja',
                                   error="Warning: Both fields are required.")

        temp_list = [x.strip() for x in temp_str.split(',') if x.strip()]
        hum_list  = [x.strip() for x in hum_str.split(',') if x.strip()]

        if len(temp_list) != 10 or len(hum_list) != 10:
            return render_template('index.html.jinja',
                                   error="Warning: Each field must have exactly 10 values.")

        # Build (10, 2) array
        data = np.column_stack((
            np.array(temp_list, dtype=float),
            np.array(hum_list, dtype=float)
        ))

        # Scale & predict
        scaled = scaler.transform(data)
        X_input = scaled.reshape((1, 10, 2))
        pred_scaled = model.predict(X_input, verbose=0)

        dummy = np.zeros((1, 2))
        dummy[0] = pred_scaled[0]
        pred = scaler.inverse_transform(dummy)[0]

        next_temp = round(float(pred[0]), 2)
        next_hum = round(float(pred[1]), 3)

        # Prepare data for charts
        labels = [f"T{i+1}" for i in range(10)] + ["Next"]

        temp_chart = data[:, 0].tolist() + [next_temp]        # 11 values
        hum_chart  = data[:, 1].tolist() + [next_hum]          # 11 values

        return render_template(
            'index.html.jinja',
            next_temp=next_temp,
            next_hum=next_hum,
            chart_labels=labels,
            temp_values=temp_chart,
            hum_values=hum_chart
        )

    except Exception as e:
        return render_template('index.html.jinja', error=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)