# app.py
import os
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score
import pickle
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalnum() and w not in stop_words]
    return ' '.join(filtered_tokens)

def train_model(chat_data, labels):
    # Drop rows with missing values in the labels
    non_nan_indices = ~pd.isnull(labels)
    chat_data = chat_data[non_nan_indices]
    labels = labels[non_nan_indices]

    tfidf_vectorizer = TfidfVectorizer(max_features=5000, preprocessor=preprocess_text)
    tfidf_matrix = tfidf_vectorizer.fit_transform(chat_data)

    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.2, random_state=42)

    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", accuracy)

    return svm_model, tfidf_vectorizer


    return svm_model, tfidf_vectorizer

def save_model(model, vectorizer, model_path="svm_model.pkl", vectorizer_path="tfidf_vectorizer.pkl"):
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)

    with open(vectorizer_path, 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

def load_model(model_path="svm_model.pkl", vectorizer_path="tfidf_vectorizer.pkl"):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return model, vectorizer

def analyze_sentiment(chat_data, model, vectorizer):
    preprocessed_chat = [preprocess_text(message) for message in chat_data]
    tfidf_matrix = vectorizer.transform(preprocessed_chat)
    predictions = model.predict(tfidf_matrix)
    positive_percentage = (sum(predictions == 1) / len(predictions)) * 100
    negative_percentage = (sum(predictions == 0) / len(predictions)) * 100
    neutral_percentage = (sum(predictions == 2) / len(predictions)) * 100
    return positive_percentage, negative_percentage, neutral_percentage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        chat_data = request.form['chat_data'].split('\n')
        positive_percentage, negative_percentage, neutral_percentage = analyze_sentiment(chat_data, svm_model, tfidf_vectorizer)
        return render_template('result.html', positive=positive_percentage, negative=negative_percentage, neutral=neutral_percentage)

if __name__ == '__main__':
    if not os.path.exists("svm_model.pkl") or not os.path.exists("tfidf_vectorizer.pkl"):
        # Training phase
        chat_df = pd.read_csv("Sentimental_analysis\custom-data-for-senti_1.csv")  # Provide path to your WhatsApp chat data CSV file
        Date = chat_df['Date'].values
        Time = chat_df['Time'].values
        Phone_no = chat_df['Phone_no'].values
        chat_data = chat_df['Message'].values
        labels = chat_df['Sentiment'].values

        svm_model, tfidf_vectorizer = train_model(chat_data, labels)
        save_model(svm_model, tfidf_vectorizer)
    else:
        # Load pre-trained model
        svm_model, tfidf_vectorizer = load_model()

    app.run(debug=True)
