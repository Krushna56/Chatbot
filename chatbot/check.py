# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

import pandas as pd

df = pd.read_csv(r"C:\Users\krush\Desktop\Internship\FUTURE_ML_03\chatbot\Dataset\customer_support_tickets.csv")
df = df.dropna(subset=['Ticket Description', 'Resolution'])
df = df.head(100)
