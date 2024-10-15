#!/bin/bash

# Load environment variables from the .env file
echo "Loading env variables..."
export $(cat .env | xargs)

# Check if the elasticsearch container exists and remove it if it does
if [ "$(docker ps -aq -f "name=elasticsearch")" ]; then
    echo "Removing elasticsearch containers..."
    docker rm -f $(docker ps -aq -f "name=elasticsearch")
    echo "Removed elasticsearch containers."
else
    echo "No elasticsearch containers to remove."
fi

# Check if the kibana container exists and remove it if it does
if [ "$(docker ps -aq -f "name=kibana")" ]; then
    docker rm -f $(docker ps -aq -f "name=kibana")
    echo "Removed kibana container."
else
    echo "No kibana container to remove."
fi

# Check if the elastic-net network exists before attempting to remove it
if docker network inspect elastic-net &>/dev/null; then
    docker network rm elastic-net
    echo "Removed network 'elastic-net'."
else
    echo "Network 'elastic-net' does not exist."
fi

# Create the network again
docker network create elastic-net

# Run the Elasticsearch container
docker run -p 127.0.0.1:9200:9200 -d --name elasticsearch --network elastic-net \
  -e ELASTIC_PASSWORD=$ELASTICSEARCH_PASSWORD \
  -e "discovery.type=single-node" \
  -e "xpack.security.http.ssl.enabled=false" \
  -e "xpack.license.self_generated.type=trial" \
  docker.elastic.co/elasticsearch/elasticsearch:8.15.2

# Check if Elasticsearch is available
until curl -u elastic:$ELASTICSEARCH_PASSWORD -s http://localhost:9200/_cluster/health | grep -q '"status":"green"'; do
  echo "Waiting for Elasticsearch to be ready..."
  sleep 5
done

echo "Elasticsearch container is running."

# Configure the Kibana password in the ES container
curl -u elastic:$ELASTICSEARCH_PASSWORD \
  -X POST \
  http://localhost:9200/_security/user/kibana_system/_password \
  -d '{"password":"'"$KIBANA_PASSWORD"'"}' \
  -H 'Content-Type: application/json'

# Run the Elasticsearch container
docker run -p 127.0.0.1:5601:5601 -d --name kibana --network elastic-net \
  -e ELASTICSEARCH_URL=http://elasticsearch:9200 \
  -e ELASTICSEARCH_HOSTS=http://elasticsearch:9200 \
  -e ELASTICSEARCH_USERNAME=kibana_system \
  -e ELASTICSEARCH_PASSWORD=$KIBANA_PASSWORD \
  -e "xpack.security.enabled=false" \
  -e "xpack.license.self_generated.type=trial" \
  docker.elastic.co/kibana/kibana:8.15.2

echo "Kibana container is running."

echo "Populating data..."

python3 ./fast_api_app/populate_data.py

echo "All data uploaded"