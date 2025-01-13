# preprocess.py
# Contains text preprocessing functions.

import re

def clean_text(text):
    """
    Clean the input text by:
    - Removing special characters
    - Lowercasing
    """
    return re.sub(r'[^a-zA-Z\s]', '', text).lower().strip()

# TODO: Add more preprocessing steps if necessary (e.g., stopword removal)
