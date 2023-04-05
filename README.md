# Kickstarter Model

This is a project library to create a model for predicting "usd_smooth_pledged_per_day" from a Kickstarter Dataset.
The model is deployed onto a Docker container and trained using a Flask API call.

## Table of Contents

### Directories
- /input_data: contains dataset used to train model
- /task_instructions: contains instructions set out for this project
- /output_data: stores outputs from model prediction for further analysis

### Python Scripts
- data_loader.py: code to ingest csv input data
- flask_server.py: code to setup flask server to trigger model training
- kickstarter_mode.py: code relevant to model creation and running

### Miscellaneous Scripts
- .gitignore: files for git to ignore
- Dockerfile: code required to set up docker container
- configuration.json: abstracted inputs used within model development
- requirements.txt: package requirements for docker
- run_model.sh: bash script to trigger creation of the docker container and train model

## Pre-Requisites
- Docker 
- Linux (Unix) Environment

## Usage
Once pre-requisites have been met:
1) Navigate to this folder within your local system.
2) Ensure you have the input csv within the input_data folder.
3) Ensure the configuration.json is correctly mapped to your input data and the rest of the configuration is all correct.
4) Run ``` run_model.sh``` either via an IDE or using ```bash run_model.sh```
5) An output_<current_epoch>.json file will be written to output_data, this contains both y_hat and y_test, so you can perform further 
downstream visualisation / analysis on the models' performance.

## Troubleshoot

- If you are getting an error relating to the port / url that is being used:
  - Ensure your firewall settings allow for http connections.
  - Ensure the port / url are not being used elsewhere. If they are:
    - Update ```Dockerfile```, ```flask_server.py```, ```run_model.sh``` to reference a new port / url based on your local system. 
- If you are getting the following error message:
  ```legacy-install-failure``` or ```subprocess-exited-with-error```
  - Update Requirements.txt and remove all versions from all packages.

## Extra Code

- If you want to copy the model from your docker container to local repository:
  - Add/ Uncomment the following code from the run_model.sh and update the destination filepath for your local system ```docker cp $containerId:/app/models/ ./models/```
- If you want to see the logs while training the model in the docker container:
  - ```docker logs <container_id> --follow```