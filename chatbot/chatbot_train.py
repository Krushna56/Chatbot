import pandas as pd
import nltk
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Load dataset from your local path
df = pd.read_csv(r"C:\Users\krush\Desktop\Internship\FUTURE_ML_03\chatbot\dataset\customer_support_tickets.csv")

# Drop rows with missing descriptions or resolutions
df = df.dropna(subset=['Ticket Description', 'Resolution'])

# Text cleaning function
def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stopwords.words('english')])

# Apply preprocessing
df['clean_text'] = df['Ticket Description'].apply(preprocess)

# Vectorize input
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['Resolution']

# Train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Save model and vectorizer
with open("chatbot_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained and saved successfully.")
