from flask import Flask, render_template, request
import pickle
import nltk
import string
 
app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("chatbot_train.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

nltk.download('punkt')
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')

def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stop_words])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_input = request.form["user_input"]
    cleaned_input = preprocess(user_input)
    vector_input = vectorizer.transform([cleaned_input])
    response = model.predict(vector_input)[0]
    return response

if __name__ == "__main__":
    app.run(debug=True)
