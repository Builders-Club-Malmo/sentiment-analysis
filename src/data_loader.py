import os

def load_data_from_directory(directory):
    texts = []
    labels = []
    for label in ["pos", "neg"]:
        dir_path = os.path.join(directory, label)
        for fname in os.listdir(dir_path):
            if fname.endswith(".txt"):
                with open(os.path.join(dir_path, fname), encoding="utf-8") as f:
                    texts.append(f.read())
                    labels.append(1 if label == "pos" else 0)
    return texts, labels