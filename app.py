from flask import Flask, render_template, request
import joblib
from keras.models import load_model 
import numpy as np
import os

app = Flask(__name__)

loaded_model = joblib.load('ridge_classifier.pkl')
# model1 = load_model('model.h5')

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get all input values from the form
        age = int(request.form['AGE'])
        sex = int(request.form.get('SEX'))  # Assuming 1 for male, 0 for female
        educ = int(request.form['EDUC'])
        ses = int(request.form['SES'])
        etiv = int(request.form['ETIV'])
        nwbv = float(request.form['NWBV'])
        asf = float(request.form['ASF'])
        mmse = int(request.form['MMSE'])

        
        
        if mmse <= 9:
            mmse = 2
        elif mmse >= 10 and mmse <= 18:
            mmse = 1  
        elif mmse >= 24 or mmse >= 19 and mmse <= 23:
            mmse = 0                            


        # Use 'CDR' for prediction
        data = np.array([[sex, age, educ, ses, etiv, nwbv, asf, mmse]])
        # print(data)

        #Predicted OUTPUT

        # cdr = model1.predict(data)
        cdr = loaded_model.predict(data)
                
        # Map the 'CDR' value to dementia stage
        if cdr == 0:
            predicted_stage = "Normal"
        elif cdr == 1:
            predicted_stage = "Mild Dementia"
        elif cdr == 2:
            predicted_stage = "Moderate Dementia"
        else:
            predicted_stage = "Unknown"

        return render_template('result.html', predicted_stage=predicted_stage)

if __name__ == '__main__':
    app.run(debug=False,port=800)
