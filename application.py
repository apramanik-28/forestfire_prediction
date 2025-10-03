from flask import Flask, request, render_template
import numpy as np
import pickle

application = Flask(__name__)
app = application

# Load ridge regressor and standard scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('home.html')  # make sure this file exists in templates/

@app.route('/predict_data', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        temperature = float(request.form.get('Temperature'))
        rh = float(request.form.get('RH'))
        ws = float(request.form.get('Ws'))  # note: case must match your HTML (Ws not WS)
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        isi = float(request.form.get('ISI'))
        classes = float(request.form.get('Classes'))
        region = float(request.form.get('Region'))

        # scale input
        input_scaled = standard_scaler.transform(
            [[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]]
        )
        result = ridge_model.predict(input_scaled)

        return render_template('home.html', result=result[0])
    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
