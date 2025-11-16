from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = load_model('model/model.h5', compile=False)
scaler = joblib.load('model/scaler.pkl')
@app.route('/')
def index():
    return render_template('index.html.jinja')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input string
        values = request.form['values']
        data = [float(x.strip()) for x in values.split(',')]
        if len(data) != 10:
            return render_template('index.html.jinja', error="⚠️ Please enter exactly 10 temperature values.")

        # Scale and reshape input
        scaled = scaler.transform(np.array(data).reshape(-1, 1))
        X_input = np.reshape(scaled, (1, 10, 1))

        # Predict next value
        pred_scaled = model.predict(X_input)
        prediction = scaler.inverse_transform(pred_scaled)[0][0]
        prediction = float(prediction)

        # Prepare chart data
        labels = [f"T{i+1}" for i in range(10)] + ["Next"]
        values_chart = data + [round(prediction, 2)]

        return render_template(
            'index.html.jinja',
            prediction=f"{prediction:.2f} °C",
            chart_labels=labels,
            chart_values=values_chart
        )
    except Exception as e:
        return render_template('index.html.jinja', error=f"❌ Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
