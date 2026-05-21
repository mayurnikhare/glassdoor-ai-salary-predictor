from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# LOAD MODEL
model = pickle.load(open('models/model.pkl', 'rb'))

@app.route('/')
def home():

    default_values = {
        'python': '0',
        'sql': '0',
        'aws': '0',
        'excel': '0',
        'tableau': '0',
        'powerbi': '0',
        'ml': '0',
        'dl': '0',
        'tensorflow': '0',
        'spark': '0',
        'hadoop': '0',
        'statistics': '0'
    }

    return render_template(
        'index.html',
        prediction_text='',
        values=default_values
    )

@app.route('/predict', methods=['POST'])
def predict():

    try:

        values = {
            'python': '1' if 'python' in request.form else '0',
            'sql': '1' if 'sql' in request.form else '0',
            'aws': '1' if 'aws' in request.form else '0',
            'excel': '1' if 'excel' in request.form else '0',
            'tableau': '1' if 'tableau' in request.form else '0',
            'powerbi': '1' if 'powerbi' in request.form else '0',
            'ml': '1' if 'ml' in request.form else '0',
            'dl': '1' if 'dl' in request.form else '0',
            'tensorflow': '1' if 'tensorflow' in request.form else '0',
            'spark': '1' if 'spark' in request.form else '0',
            'hadoop': '1' if 'hadoop' in request.form else '0',
            'statistics': '1' if 'statistics' in request.form else '0'
        }

        # CHECK IF NO SKILL SELECTED
        total_selected = sum(
            int(value)
            for value in values.values()
        )

        if total_selected == 0:

            return render_template(
                'index.html',
                prediction_text='Please select at least one skill',
                values=values
            )

        # FEATURE ARRAY
        features = np.array([[
            int(values['python']),
            int(values['sql']),
            int(values['aws']),
            int(values['excel']),
            int(values['tableau']),
            int(values['powerbi']),
            int(values['ml']),
            int(values['dl']),
            int(values['tensorflow']),
            int(values['spark']),
            int(values['hadoop']),
            int(values['statistics'])
        ]])

        # PREDICTION
        prediction = model.predict(features)

        output = round(prediction[0], 2)

        return render_template(
            'index.html',
            prediction_text=f'Estimated Salary: ${output}K',
            values=values
        )

    except Exception as e:

        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}',
            values={}
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)