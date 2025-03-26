# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_preprocessing import clean_text
from src.feature_extraction import extract_features
from src.model_training import train_and_save_model, evaluate_model
from src.utils import load_model
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import os

# Load Dataset
data_path = "data/IMDB Dataset.csv"
df = pd.read_csv(data_path)

# Check Data
print("Sample Data:")
print(df.head())

# Data Preprocessing
df['cleaned_review'] = df['review'].apply(clean_text)

# Split Data
X = df['cleaned_review']
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Extraction
X_train_tfidf, X_test_tfidf, vectorizer = extract_features(X_train, X_test)

# Model Paths
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Logistic Regression Model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model_path = os.path.join(models_dir, "logistic_regression.pkl")
train_and_save_model(logistic_model, X_train_tfidf, y_train, logistic_model_path)
logistic_model_loaded = load_model(logistic_model_path)
print("\nLogistic Regression Results:")
evaluate_model(logistic_model_loaded, X_test_tfidf, y_test)

# Naive Bayes Model
nb_model = MultinomialNB()
nb_model_path = os.path.join(models_dir, "naive_bayes.pkl")
train_and_save_model(nb_model, X_train_tfidf, y_train, nb_model_path)
nb_model_loaded = load_model(nb_model_path)
print("\nNaive Bayes Results:")
evaluate_model(nb_model_loaded, X_test_tfidf, y_test)

# Support Vector Machine (SVM) Model
svm_model = SVC(kernel='linear')
svm_model_path = os.path.join(models_dir, "svm_model.pkl")
train_and_save_model(svm_model, X_train_tfidf, y_train, svm_model_path)
svm_model_loaded = load_model(svm_model_path)
print("\nSupport Vector Machine Results:")
evaluate_model(svm_model_loaded, X_test_tfidf, y_test)

# Sample Prediction Function
def predict_review(review, model_path, vectorizer):
    cleaned_review = clean_text(review)
    vectorized_review = vectorizer.transform([cleaned_review])
    model = load_model(model_path)
    prediction = model.predict(vectorized_review)[0]
    return "Positive" if prediction == 1 else "Negative"

# Test on a Sample Review
sample_review = "The movie was absolutely fantastic and I loved every part of it!"
result = predict_review(sample_review, logistic_model_path, vectorizer)
print(f"\nSample Review Prediction (Logistic Regression): {result}")
