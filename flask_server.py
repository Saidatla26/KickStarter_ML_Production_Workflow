from data_loader import DataLoader
from kickstarter_model import KickstarterPredictor
from flask import Flask, request, jsonify

# Creates a Flask application object in the current Python module.
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def train_and_predict() -> object:
    """
    Uses the app.route() decorator to let Flask know to trigger this function when the url /predict is used.
    This function is called by Flask to train and predict our data for the ML model. If we have configured
    the ML app to use an existing model and one exists then we will not re-generate the model again and directly
    predict the data.
    """
    # Get the JSON configuration input from the http request.
    configuration = request.get_json()

    # Define the DataLoader object using input from the configuration file.
    data_loader = DataLoader(configuration['input_data_attributes'].get("data_path"),
                             configuration['input_data_attributes'].get("index_col"),
                             configuration['input_data_attributes'].get("y_col"),
                             configuration['input_data_attributes'].get("test_size"),
                             configuration['input_data_attributes'].get("random_state"))

    # Retrieve the training/testing data from the DataLoader load_data function.
    x_train, x_test, y_train, y_test = data_loader.load_data()

    # Define KickStarterPredictor object.
    model = KickstarterPredictor()

    # Build the model within the object, passing in attributes from our configuration.
    model.build_model(configuration['model_build_attributes'].get("optimizer"),
                    configuration['model_build_attributes'].get("loss"),
                    configuration['model_build_attributes'].get("metrics"))

    # Train the model using data we pre-processed above, and attributes from our configuration.
    model.train(x_train, y_train, configuration['model_train_attributes'].get("batch_size"),
                configuration['model_train_attributes'].get("epochs"),
                configuration['model_train_attributes'].get("validation_split"))

    # Save the model locally, using the name provided within the configuration file.
    model.model.save('trained_model')

    # Predict our output, and re-format so it is a list of floats.
    y_pred = model.predict(x_test)
    y_pred = [float(i) for i in y_pred]

    # Convert y_test to list, this allows us to return the response as a JSON format.
    y_test_list = y_test.tolist()

    # Copy the list generated above into a dictionary, this allows for us to return the results in JSON format.
    response = {'prediction': y_pred, 'y_test': y_test_list}

    # Returns JSON serialized data as a response to URL call.
    return jsonify(response)


@app.route('/test')
def hello_world() -> str:
    """
    Uses the app.route() decorator to let Flask know to trigger this function when URL /test is used.
    The function is a simple test to return the string "Hello, World!" if the API call is made.
    """
    # Returns the string "Hello, World!"
    return 'Hello, World!'


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5500, debug=True)