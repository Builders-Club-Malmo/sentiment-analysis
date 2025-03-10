# train.py
# This script will be used to train your sentiment analysis model.

# TODO: Import necessary libraries (e.g., pandas, scikit-learn, joblib)

def evaluate_model(y_true, y_pred):
    """
    Prints evaluation metrics for the model.
    """
    # TODO: Implement evaluation logic (e.g., accuracy, precision, recall)
    pass

def train_model(data_path='data/imdb_sample.csv', model_output='model.joblib'):
    """
    Step 1: Load your dataset
    Step 2: Preprocess your text data
    Step 3: Vectorize the text (e.g., using TfidfVectorizer)
    Step 4: Train a machine learning model (e.g., Logistic Regression)
    Step 5: Evaluate the model
    Step 6: Save the trained model to model.joblib
    """
    # TODO: Load dataset using pandas
    # TODO: Split data into training and testing sets
    # TODO: Vectorize text data using TfidfVectorizer
    # TODO: Train a classifier (e.g., Logistic Regression)
    # TODO: Evaluate the model using the evaluate_model function
    # TODO: Save the trained model and vectorizer using joblib
    pass

if __name__ == '__main__':
    train_model()
