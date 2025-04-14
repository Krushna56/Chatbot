from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

# Load trained chatbot model and vectorizer
try:
    model = joblib.load('model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    raise

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_input = request.json["msg"]
    # Vectorize the input text
    input_vectorized = vectorizer.transform([user_input])
    response = model.predict(input_vectorized)[0]
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)


