import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


class DataLoader:
    """
    A class that represents a simple csv data loader and pre-processor.

    Attributes:
        data_path (string): The path to the data file.
        index_col (string): The index column for the data in the file.
        y_col (string): The column you want to predict.
        test_size (float): The size of the data split, can be an integer or float.
        random_state (float): The seed for the random generator to allow for reproducable results.

    Methods:
       load_data: Retrieves and splits the data from a csv file.
    """

    def __init__(self, data_path: str, index_col: str, y_col: str, test_size: float, random_state: float) -> None:
        """
        Initializes the DataLoader instance.

        Args:
            data_path (string): The path to the data file.
            index_col (string): The index column for the data in the file.
            y_col (string): The column you want to predict.
            test_size (float): The size of the data split, can be an integer or float.
            random_state (float): The seed for the random generator to allow for reproducable results.
        """
        # Assign class attributes using ingested parameters
        self.data_path = data_path
        self.index_col = index_col
        self.y_col = y_col
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads and splits the KickStarter data used to train and test the model.

        Returns:
            x_train (np.ndarray): The subset of the data containing independent variables used to train the model.
            x_test (np.ndarray): The remainder of the independent variables used to make predictions.
            y_train (np.ndarray): The subset of the data containing dependent variables used to train the model.
            y_test (np.ndarray): The remainder of the dependent variables used to test the accuracy of predictions.
        """
        # Load data from csv file
        train_test_df = pd.read_csv(self.data_path, index_col=self.index_col)

        # Assign test size for processing input data
        test_size = self.test_size

        # Assign random state for generating train/test data
        random_state = self.random_state

        # Drop all NA values from the target column
        train_test_df = train_test_df.dropna(subset=[self.y_col])

        # Separate the data into features and target
        x = train_test_df.drop(self.y_col, axis=1)
        y = train_test_df[self.y_col]

        # Split the data into training/test data for both feature and target columns
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state)

        return x_train, x_test, y_train, y_test

