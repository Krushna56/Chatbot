import pandas as pd
import nltk
nltk.data.path.append("C:/Users/krush/AppData/Roaming/nltk_data")
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords

print("🔁 Downloading NLTK resources...")
nltk.download('punkt_tab')
nltk.download('stopwords')

print("📂 Loading dataset...")
data = pd.read_csv('training_data.csv')  # Create this CSV with your chatbot training data
X = data['user_input']
y = data['response']

print("🧹 Preprocessing text...")
def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stopwords.words('english')])

X = X.apply(preprocess)

print("📊 Vectorizing...")
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

print("🧠 Training model...")
model = MultinomialNB()
model.fit(X_vectorized, y)

print("💾 Saving model and vectorizer...")
joblib.dump(model, 'model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

print("✅ All done!")
print("Model and vectorizer saved successfully.")
# The model is trained and saved as 'model.joblib' and the vectorizer as 'vectorizer.joblib'.