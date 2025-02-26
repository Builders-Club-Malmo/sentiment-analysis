from download_extract import ensure_data_directory
from data_loader import load_data_from_directory
from preprocess import preprocess_texts
from train import train_model
from predict import predict_sentiment
from utils import evaluate_model
from sklearn.model_selection import train_test_split

# Set the data directory manually
data_dir = "aclImdb"

# Ensure the data directory exists
ensure_data_directory(data_dir)

# Load data from the provided directory
train_texts, train_labels = load_data_from_directory("aclImdb/train")
test_texts, test_labels = load_data_from_directory("aclImdb/test")

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)

train_padded, val_padded, test_padded, tokenizer = preprocess_texts(train_texts, val_texts, test_texts)

# Pass the tokenizer to the train_model function
model = train_model(train_padded, train_labels, val_padded, val_labels, tokenizer)

accuracy, report = evaluate_model(model, test_padded, test_labels)
print(f"Test Accuracy: {accuracy:.4f}")
print(report)

sample_review = "This movie was absolutely fantastic! The performances were stunning."
prediction, sentiment = predict_sentiment(model, tokenizer, sample_review)
print(f"Sentiment Score: {prediction:.4f} ({sentiment})")