# Define python as the parent image.
FROM python:latest
# Required for bugfix with package installations.
ENV DOCKER_BUILDKIT=0
# Set working directory.
WORKDIR /app
# Copy requirements into docker container.
COPY requirements.txt .
# Run pip install on packages listed within requirements.txt.
RUN pip install --trusted-host pypi.python.org -r requirements.txt
# Copy the rest of the app code into the container.
COPY . .
# Set the environment variables for Flask, including the app python file and whether to enable debugging (only for dev).
ENV FLASK_APP=flask_server.py
#ENV FLASK_DEBUG=True
# Expose the port used by Flask.
EXPOSE 5500
# Run the Flask app when the container starts, on the specified host and port.
CMD ["flask", "run", "--host=0.0.0.0", "--port=5500"]


