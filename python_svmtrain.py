import pickle
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

df = df[['v1','v2']]

df.rename(columns={
    'v1':'target',
    'v2':'text'
}, inplace=True)

df.drop_duplicates(inplace=True)

encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

ps = PorterStemmer()

def transform_text(text):

    text = text.lower()

    text = nltk.word_tokenize(text)

    y = []

    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        if word not in stopwords.words('english'):
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)

df['transformed_text'] = df['text'].apply(transform_text)

tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df['transformed_text'])

y = df['target'].values

model = LinearSVC()

model.fit(X, y)

pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))

print("SVM model saved successfully!")