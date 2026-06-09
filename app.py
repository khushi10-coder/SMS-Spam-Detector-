import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# =========================
# Custom CSS
# =========================

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

h1 {
    text-align:center;
}

.stButton > button {
    width:100%;
    height:50px;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# Load Model
# =========================

@st.cache_resource
def load_models():
    tfidf = pickle.load(open("vectorizer.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
    return tfidf, model

tfidf, model = load_models()

# =========================
# Session History
# =========================

if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# NLTK
# =========================

ps = PorterStemmer()

# =========================
# Sidebar
# =========================

with st.sidebar:

    st.header("📊 Model Information")

    st.write("**Algorithm:** Linear SVM")
    st.write("**Vectorizer:** TF-IDF")

    st.divider()

    st.write("### Performance")

    st.write("Accuracy: 97.78%")
    st.write("Precision: 96.75%")
    st.write("Recall: 86.23%")

    st.divider()

    st.metric("Dataset Size", "5169")
    st.metric("Ham Messages", "4516")
    st.metric("Spam Messages", "653")

# =========================
# Text Preprocessing
# =========================

def transform_text(text):

    text = text.lower()

    text = nltk.word_tokenize(text)

    y = []

    # Remove special characters
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

# =========================
# Main UI
# =========================

st.title("📩 SMS Spam Detector")

st.write(
    "Detect whether a message is Spam or Ham using Machine Learning."
)

# Sample Messages

sample = st.selectbox(
    "Try Sample Messages",
    [
        "",
        "Congratulations! You won ₹50,000. Claim now.",
        "URGENT! Your bank account has been blocked.",
        "Hey, are we meeting tomorrow at 5 PM?",
        "Mom called. Please call her back."
    ]
)

input_sms = st.text_area(
    "Enter Message",
    value=sample,
    height=150
)

# =========================
# Prediction
# =========================

if st.button("🔍 Predict"):

    if input_sms.strip() == "":
        st.warning("Please enter a message.")

    else:

        # Statistics

        st.subheader("📊 Message Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Characters", len(input_sms))

        with col2:
            st.metric("Words", len(input_sms.split()))

        # Transform

        transformed_sms = transform_text(input_sms)

        # Vectorize

        vector_input = tfidf.transform([transformed_sms])

        # Predict

        result = model.predict(vector_input)[0]

        prediction_text = "Spam" if result == 1 else "Ham"

        st.session_state.history.append({
            "Message": input_sms[:60],
            "Prediction": prediction_text
        })

        st.divider()

        st.subheader("🎯 Prediction Result")

        if result == 1:
            st.error("🚨 Spam Message Detected")
        else:
            st.success("✅ Ham Message")

        st.divider()

        st.subheader("🔍 Processed Text")

        st.code(transformed_sms)

# =========================
# Prediction History
# =========================

st.divider()

st.subheader("📝 Prediction History")

if len(st.session_state.history) > 0:
    st.dataframe(
        st.session_state.history,
        use_container_width=True
    )
else:
    st.info("No predictions yet.")