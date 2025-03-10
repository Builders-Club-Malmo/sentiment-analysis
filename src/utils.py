# utils.py
# Contains helper functions for the project.

import pandas as pd

def load_data(file_path):
    """
    Load the dataset from the given file path.
    """
    return pd.read_csv(file_path)
