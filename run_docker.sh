#!/bin/bash
# run_docker.sh
docker build -t cuerdas-maldacena .
docker run -it --rm \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/boundary:/app/boundary \
  cuerdas-maldacena \
  bash