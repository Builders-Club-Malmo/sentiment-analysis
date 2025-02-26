import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# Load the trained model
model = load_model("sentiment_model.h5")

# Load the tokenizer
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

def predict_sentiment(review, max_length=500):
    sequence = tokenizer.texts_to_sequences([review])
    padded = np.array(pad_sequences(sequence, maxlen=max_length, padding="post", truncating="post"))
    prediction = model.predict(padded)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return float(prediction), sentiment

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data['review']
    prediction, sentiment = predict_sentiment(review)
    return jsonify({'prediction': prediction, 'sentiment': sentiment})

if __name__ == "__main__":
    app.run(debug=True)