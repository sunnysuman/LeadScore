
import numpy as np
from flask import Flask, request, jsonify, render_template,url_for
import pickle
import os

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))



#picfolder=os.path.join('static','images')
#app.config['UPLOAD_FOLDER']=picfolder

@app.route('/')
def home():
    #pic=os.path.join(app.config['UPLOAD_FOLDER'],'4.png')
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    
    #For rendering results on HTML GUI
    
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Lead Score should be {}'.format(output))




'''

@app.route('/predict',methods=['GET','POST'])
def predict():
    lead=request.form.get("Lead Number")
    source=request.form.get("Lead Source")

    prediction = model.predict(lead,source)
    #output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Lead Score should be $ {}'.format(output))
'''

if __name__ == "__main__":
    app.run(debug=True)