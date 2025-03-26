from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Train Model and Save
def train_and_save_model(model, X_train, y_train, model_path):
    model.fit(X_train, y_train)
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
