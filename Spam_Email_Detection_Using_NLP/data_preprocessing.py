import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

required_packages = ['punkt', 'stopwords', 'wordnet']


for pkg in required_packages:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if pkg == 'punkt' else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
vectorizor = TfidfVectorizer()

df = pd.read_csv("dataset/enron_spam_data.csv")

df["text"] = df["Subject"].fillna(" ") + df["Message"].fillna(" ")


def clean_text(text):

    tokens = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text) if word.isalnum() and word.lower() not in stop_words]

    return " ".join(tokens)

label_map = {"ham": 0 , "spam" : 1}
df["labels"] = df["Spam/Ham"].map(label_map)
df["text"] = df["text"].apply(clean_text)

vector = vectorizor.fit_transform(df["text"])

