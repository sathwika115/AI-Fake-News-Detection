import os
import re
import joblib
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="wide"
)


# =====================================================
# CSS / FRONTEND DESIGN
# =====================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 20% 20%,
            rgba(99,102,241,0.20),
            transparent 30%
        ),
        radial-gradient(
            circle at 80% 80%,
            rgba(168,85,247,0.18),
            transparent 30%
        ),
        #0f172a;

    color: white;
}

.main-title {
    text-align: center;
    font-size: 45px;
    font-weight: 800;
    margin-top: 20px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 35px;
}

.result-real {
    background: rgba(34,197,94,0.15);
    border: 1px solid #22c55e;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
}

.result-fake {
    background: rgba(239,68,68,0.15);
    border: 1px solid #ef4444;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# FILE
# =====================================================

DATASET = "fake_news.csv"

MODEL_FILE = "fake_news_model.pkl"

VECTORIZER_FILE = "tfidf_vectorizer.pkl"


# =====================================================
# TEXT CLEANING
# =====================================================

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =====================================================
# TRAIN MODEL
# =====================================================

@st.cache_resource
def train_model():

    if not os.path.exists(DATASET):

        st.error(
            "fake_news.csv not found. "
            "Place it in the same folder as app.py."
        )

        st.stop()


    # Read CSV
    df = pd.read_csv(DATASET)


    # Check columns
    if "text" not in df.columns:

        st.error(
            "CSV must contain a 'text' column."
        )

        st.stop()


    if "label" not in df.columns:

        st.error(
            "CSV must contain a 'label' column."
        )

        st.stop()


    # Remove empty rows
    df = df.dropna(
        subset=["text", "label"]
    )


    # Clean text
    df["text"] = df["text"].apply(
        clean_text
    )


    # Clean labels
    df["label"] = (
        df["label"]
        .astype(str)
        .str.upper()
        .str.strip()
    )


    # Convert labels
    label_map = {
        "FAKE": 0,
        "REAL": 1,
        "0": 0,
        "1": 1
    }


    df["target"] = df["label"].map(
        label_map
    )


    # Remove invalid labels
    df = df.dropna(
        subset=["target"]
    )


    X = df["text"]

    y = df["target"].astype(int)


    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )


    # TF-IDF
    vectorizer = TfidfVectorizer(

        stop_words="english",

        max_features=50000,

        ngram_range=(1, 2)
    )


    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )


    X_test_tfidf = vectorizer.transform(
        X_test
    )


    # Machine Learning model
    model = LogisticRegression(
        max_iter=1000
    )


    model.fit(
        X_train_tfidf,
        y_train
    )


    # Prediction
    predictions = model.predict(
        X_test_tfidf
    )


    # Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )


    # Save trained model
    joblib.dump(
        model,
        MODEL_FILE
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_FILE
    )


    return (
        model,
        vectorizer,
        accuracy,
        len(df)
    )


# =====================================================
# LOAD MODEL
# =====================================================

model, vectorizer, accuracy, dataset_size = train_model()


# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="main-title">📰 AI Fake News Detector</div>',
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="subtitle">
    AI-powered Fake News Detection using
    Machine Learning and NLP
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("⚙️ Project Information")

    st.write("### 🤖 AI Technology")

    st.write(
        "Machine Learning + NLP"
    )

    st.write("### 🧠 Algorithm")

    st.write(
        "Logistic Regression"
    )

    st.write("### 🔤 Feature Extraction")

    st.write(
        "TF-IDF Vectorization"
    )

    st.write("### 📊 Dataset")

    st.write(
        f"{dataset_size} articles"
    )

    st.write("### 🎯 Model Accuracy")

    st.write(
        f"{accuracy * 100:.2f}%"
    )

    st.divider()

    st.info(
        "No API key is required."
    )


# =====================================================
# NEWS INPUT
# =====================================================

st.subheader(
    "🔍 Check a News Article"
)


news_text = st.text_area(

    "Paste your news article or headline:",

    height=250,

    placeholder=
    "Enter or paste news here..."
)


# =====================================================
# BUTTONS
# =====================================================

col1, col2 = st.columns(2)


with col1:

    detect_button = st.button(
        "🔍 Detect News",
        use_container_width=True
    )


with col2:

    clear_button = st.button(
        "🗑️ Clear",
        use_container_width=True
    )


# =====================================================
# CLEAR
# =====================================================

if clear_button:

    st.rerun()


# =====================================================
# PREDICTION
# =====================================================

if detect_button:

    if not news_text.strip():

        st.warning(
            "⚠️ Please enter a news article."
        )


    elif len(news_text.strip()) < 10:

        st.warning(
            "⚠️ Please enter more text."
        )


    else:

        with st.spinner(
            "🤖 AI is analyzing the news..."
        ):

            # Clean input
            cleaned_news = clean_text(
                news_text
            )


            # TF-IDF
            transformed_text = vectorizer.transform(
                [cleaned_news]
            )


            # Prediction
            prediction = model.predict(
                transformed_text
            )[0]


            # Probability
            probabilities = model.predict_proba(
                transformed_text
            )[0]


            fake_probability = (
                probabilities[0] * 100
            )


            real_probability = (
                probabilities[1] * 100
            )


            if prediction == 0:

                result = "FAKE"

                confidence = fake_probability

            else:

                result = "REAL"

                confidence = real_probability


        st.divider()


        # =================================================
        # RESULT
        # =================================================

        if result == "FAKE":

            st.markdown(
                f"""
                <div class="result-fake">

                    <h1>❌ FAKE NEWS</h1>

                    <h3>
                    Confidence: {confidence:.2f}%
                    </h3>

                    <p>
                    The AI model classified this
                    news as likely fake.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


        else:

            st.markdown(
                f"""
                <div class="result-real">

                    <h1>✅ REAL NEWS</h1>

                    <h3>
                    Confidence: {confidence:.2f}%
                    </h3>

                    <p>
                    The AI model classified this
                    news as likely real.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # PROBABILITY
        # =================================================

        st.subheader(
            "📊 Prediction Analysis"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "❌ Fake Probability",
                f"{fake_probability:.2f}%"
            )

            st.progress(
                int(fake_probability)
            )


        with col2:

            st.metric(
                "✅ Real Probability",
                f"{real_probability:.2f}%"
            )

            st.progress(
                int(real_probability)
            )


        # =================================================
        # MODEL DETAILS
        # =================================================

        st.divider()

        st.subheader(
            "🧠 AI Model Information"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Algorithm",
                "Logistic Regression"
            )


        with col2:

            st.metric(
                "Features",
                "TF-IDF"
            )


        with col3:

            st.metric(
                "Accuracy",
                f"{accuracy * 100:.2f}%"
            )


# =====================================================
# FOOTER
# =====================================================

st.markdown(
    """
    <div class="footer">

    📰 AI-Based Fake News Detection System

    <br><br>

    Python • Streamlit • Scikit-learn • NLP

    <br><br>

    ⚠️ The prediction is based on patterns
    learned from the training dataset.

    </div>
    """,
    unsafe_allow_html=True
)