#!/bin/bash
docker run --gpus all --rm -u $(id -u ${USER}):$(id -g ${USER}) -v /data:/data -v $PWD:/pwd -w /pwd -i -t tf_docker phire-train

