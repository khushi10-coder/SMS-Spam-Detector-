import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, precision_score, recall_score

# ==========================
# NLTK Downloads
# ==========================

nltk.download('punkt_tab')
nltk.download('stopwords')

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only useful columns
df = df[['v1', 'v2']]

# Rename columns
df.rename(columns={
    'v1': 'target',
    'v2': 'text'
}, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Encode target
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# ==========================
# Text Preprocessing
# ==========================

ps = PorterStemmer()

def transform_text(text):

    text = text.lower()

    text = nltk.word_tokenize(text)

    y = []

    # Keep only alphanumeric words
    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    # Remove stopwords
    for word in text:
        if word not in stopwords.words('english'):
            y.append(word)

    text = y[:]
    y.clear()

    # Stemming
    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)

# Create transformed text column
df['transformed_text'] = df['text'].apply(transform_text)

# ==========================
# TF-IDF
# ==========================

tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df['transformed_text'])

y = df['target'].values

print("Dataset Shape:", X.shape)

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=2
)

print("Training Samples:", X_train.shape[0])
print("Testing Samples :", X_test.shape[0])

# ==========================
# Models
# ==========================

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC()
}

# ==========================
# Training & Evaluation
# ==========================

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n" + "="*50)
    print(name)
    print("="*50)

    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred), 4))
    print("Recall   :", round(recall_score(y_test, y_pred), 4))