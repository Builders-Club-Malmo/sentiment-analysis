import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import re

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word.lower() not in stop_words])
    return text

def preprocess_texts(train_texts, val_texts, test_texts, max_length=500):
    train_texts = [clean_text(t) for t in train_texts]
    val_texts = [clean_text(t) for t in val_texts]
    test_texts = [clean_text(t) for t in test_texts]

    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)

    train_sequences = tokenizer.texts_to_sequences(train_texts)
    val_sequences = tokenizer.texts_to_sequences(val_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)

    train_padded = np.array(pad_sequences(train_sequences, maxlen=max_length, padding="post", truncating="post"))
    val_padded = np.array(pad_sequences(val_sequences, maxlen=max_length, padding="post", truncating="post"))
    test_padded = np.array(pad_sequences(test_sequences, maxlen=max_length, padding="post", truncating="post"))

    return train_padded, val_padded, test_padded, tokenizer