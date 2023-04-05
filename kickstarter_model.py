import numpy as np
import tensorflow as tf


class KickstarterPredictor:
    """
    A class that represents the KickStarter Model.

    Attributes:
        model (None): Creating and setting a default value of model variable to None.

    Methods:
       build_model: Compiles the model using specified parameters.
       train: Trains the model using our training data, and specified parameters.
       predict: Generates y_hat (predicted output) using data we provide.
    """

    def __init__(self):
        """
        Initializes the KickStarterPredictor instance. This includes setting a default of None to the model attribute.
        """
        # Sets Class Attribute model to None when a new instance is instantiated.
        self.model = None

    def build_model(self, optimizer: str, loss: str, metrics: str) -> callable:
        """
        Builds the KickStarterPredictor model using attributes we have configured.

        Args:
            optimizer (string): The class/function that will be used to change the attributes within the model to
                                minimize loss.
            loss (string): The class name of the function being applied from keras.losses, which is used to compute
                           the quantity that a model should seek to minimize during training.
            metrics (string): The class name of the function being applied from keras.metrics, which is used to judge
                              the performance of your model.
        """
        # Defines the model.
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='tanh'),
            tf.keras.layers.Dense(256, activation='tanh'),
            tf.keras.layers.Dense(1)
        ])

        # Compile model with specified optimizer, loss and metric functions.
        self.model.compile(optimizer=optimizer,
                           loss=eval(f"tf.keras.losses.{loss}()"),
                           metrics=eval(f"tf.keras.losses.{metrics}()"))

    def train(self, x_train: np.ndarray, y_train: np.ndarray, batch_size: int, epochs: int, validation_split: float) \
            -> None:
        """
        Trains the KickStarterPredictor model using data and attributes we have configured.

        Args:
            x_train (np.ndarray): The subset of the data containing independent variables used to train the model.
            y_train (np.ndarray): The subset of the data containing dependent variables used to train the model.
            batch_size (int): The number of samples process before model is updated.
            epochs (int): The number of iterations through the training data.
            validation_split (float): Value between 0 & 1. Fraction of the training data to be used as validation data.
        """
        # Train the model, using our separated training data, and other attributes from the configuration file.
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def predict(self, x_test) -> np.ndarray:
        """
        Uses the model with inputted data and generates predicted output (y_hat).

        Args:
            x_test (np.ndarray): The remainder of the independent variables used to make predictions.
        """
        # Return the predicted output from the model for the inputted data.
        return self.model.predict(x_test)
