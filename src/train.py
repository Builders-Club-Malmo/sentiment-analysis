import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

def train_model(train_padded, train_labels, val_padded, val_labels, tokenizer):
    model = Sequential([
        Embedding(input_dim=10000, output_dim=128, input_length=500),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.summary()
    model.fit(train_padded, np.array(train_labels), validation_data=(val_padded, np.array(val_labels)), epochs=10, batch_size=128)

    # Save the model
    model.save("sentiment_model.h5")

    # Save the tokenizer
    with open("tokenizer.pickle", "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return model