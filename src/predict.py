from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

def predict_sentiment(model, tokenizer, review, max_length=500):
    sequence = tokenizer.texts_to_sequences([review])
    padded = np.array(pad_sequences(sequence, maxlen=max_length, padding="post", truncating="post"))
    prediction = model.predict(padded)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return prediction, sentiment

if __name__ == "__main__":
    # Load the trained model
    model = load_model("sentiment_model.h5")

    # Load the tokenizer
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)

    # Take input string from the user
    review = input("Enter a review: ")

    # Predict sentiment
    prediction, sentiment = predict_sentiment(model, tokenizer, review)
    print(f"Prediction: {prediction}, Sentiment: {sentiment}")