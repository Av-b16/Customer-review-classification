import re
import string

# Text Cleaning and Preprocessing
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r"\d+", "", text)  # Remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text
