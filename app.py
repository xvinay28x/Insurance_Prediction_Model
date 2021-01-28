from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open("pickle_file.pkl", "rb"))
app = Flask(__name__)

@app.route('/')
def man():
    return render_template("home.html")


@app.route('/predict', methods=['POST'])
def home():
    data = request.form['a']
    arr = np.array([[data]],dtype='float64')
    pred = model.predict(arr)
    output = round(pred[0], 2)
    return render_template("home.html",answer = output)
if __name__ == "__main__":
    app.run(debug=True)