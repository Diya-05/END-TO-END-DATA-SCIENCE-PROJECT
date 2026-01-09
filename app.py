from flask import Flask, request
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return open("index.html").read()

@app.route('/predict', methods=['POST'])
def predict():
    study = float(request.form['study'])
    attendance = float(request.form['attendance'])

    prediction = model.predict([[study, attendance]])

    return "PASS" if prediction[0] == 1 else "FAIL"

if __name__ == "__main__":
    app.run(debug=True)
