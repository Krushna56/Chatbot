from flask import Flask, render_template, request, jsonify
import pickle
import nltk
import string

# Load model and vectorizer
model = pickle.load(open("chatbot_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# NLTK setup
nltk.data.path.append("C:/Users/krush/AppData/Roaming/nltk_data")
nltk.download('punkt')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Flask app
app = Flask(__name__)

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stop_words])

# Route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def chatbot_response():
    user_input = request.form['msg']
    clean_input = preprocess(user_input)
    vector_input = vectorizer.transform([clean_input])
    response = model.predict(vector_input)[0]
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
