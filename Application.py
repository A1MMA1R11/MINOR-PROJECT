# import libraries
import numpy as np
import pandas as pd
from flask import Flask, request,jsonify, render_template
import pickle

# Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('Breast_Cancer.sav', 'rb'))


# default page of our web-app
@app.route('/')
def home():
    return render_template('index1.html')

# To use the predict button in our web-app
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    try:
        int_features = [float(x) for x in request.form.values()]
    except ValueError:
        return render_template('index1.html', prediction_text="Invalid input: Please enter numeric values.")
    final_features = [np.array(int_features)]
    features_name = ['mean_radius', 'mean_texture', 'mean_perimeter']
    df = pd.DataFrame(final_features, columns=features_name)
    output = model.predict(df.values)[0]

    print(output)

    if output == 0:
        res_val = "no breast cancer"
    else:
        res_val = "breast cancer"
    return render_template('index1.html', prediction_text='Patient has :{}'.format(res_val))


if __name__ == "__main__":
    app.run(debug=True)