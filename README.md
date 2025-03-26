Project Overview:-
This project aims to classify customer reviews as positive or negative using machine learning models. It demonstrates text classification through the following steps:

1. Text Preprocessing
2. Feature Extraction using TF-IDF
3. Model Training with Logistic Regression, Naive Bayes, and SVM
4. Evaluation of Models
5. Saving and Loading Models
6. Predicting Sentiments of New Reviews

Dataset Information:-
Dataset Used: IMDB Reviews Dataset
The dataset contains 50,000 movie reviews labeled as positive or negative.
Download and place the dataset in the data/ directory.

Installation and Setup
1. Clone the Repository

git clone https://github.com/yourusername/customer_review_classification.git
cd customer_review_classification

2. Create Virtual Environment

# Create virtual environment
python -m venv venv

# Activate environment (Linux/Mac)
source venv/bin/activate

# OR (Windows)
venv\Scripts\activate

3. Install Required Packages

pip install -r requirements.txt

4. Run the Main Script

python main.py

Requirements:-
pandas
numpy
scikit-learn
nltk
pickle-mixin

How to Run the Project
1️⃣ Run Main Script
python main.py

2️⃣ Jupyter Notebook for Analysis
If you prefer to analyze and experiment with the code in a Jupyter Notebook:
jupyter notebook
Open customer_review_classification.ipynb from the notebooks/ folder.
