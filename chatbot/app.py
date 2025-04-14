from flask import Flask, render_template, request, jsonify
import pandas as pd

from chatbot_train import model

# Load trained chatbot model
model = joblib.load(r"C:\Users\krush\Desktop\Internship\FUTURE_ML_03\chatbot\chatbot_model.pkl")

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


