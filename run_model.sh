echo "Creating docker container."

docker build -t kickstarter_ml .

echo "Running docker container."

container_id=$(docker run -d -p 5500:5500 kickstarter_ml)

echo "Waiting for container to run."

sleep 20

echo "Running model and outputting to json file named with current epoch datetime."

current_epoch_time=$(date +%s)

curl -X POST http://localhost:5500/predict -H "Content-Type: application/json" -d @./configuration.json > ./output_data/output_$current_epoch_time.json

# Code to copy over model files into local repository, need to manually update the local repository destination
# docker cp $containerId:/app/models/ ./models/