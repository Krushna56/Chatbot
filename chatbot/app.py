from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd 
import nltk
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 


# Load trained chatbot model
model = joblib.load("chatbot_train.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_input = request.json["msg"]
    response = model.predict([user_input])[0]
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
