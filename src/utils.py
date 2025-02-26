from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, test_padded, test_labels):
    test_predictions = (model.predict(test_padded) > 0.5).astype("int32")
    accuracy = accuracy_score(test_labels, test_predictions)
    report = classification_report(test_labels, test_predictions)
    return accuracy, report