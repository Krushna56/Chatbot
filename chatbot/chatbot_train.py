import pandas as pd
import nltk
nltk.data.path.append("C:/Users/krush/AppData/Roaming/nltk_data")
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords

print("🔁 Downloading NLTK resources...")
nltk.download('punkt_tab')
nltk.download('stopwords')

print("📂 Loading dataset...")
df = pd.read_csv(r"C:\Users\krush\Desktop\Internship\FUTURE_ML_03\chatbot\Dataset\customer_support_tickets.csv")
df = df.dropna(subset=['Ticket Description', 'Resolution'])
df = df.head(500)  # for testing

print("🧹 Preprocessing text...")
def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stopwords.words('english')])

df['clean_text'] = df['Ticket Description'].apply(preprocess)

print("📊 Vectorizing...")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['Resolution']

print("🧠 Training model...")
model = LogisticRegression()
model.fit(X, y)

print("💾 Saving model and vectorizer...")
with open("chatbot_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ All done!")
print("Model and vectorizer saved successfully.")
# The model is trained and saved as 'chatbot_model.pkl' and the vectorizer as 'vectorizer.pkl'.