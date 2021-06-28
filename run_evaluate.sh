#!/bin/bash
docker run --gpus all --rm -u $(id -u ${USER}):$(id -g ${USER}) -v /home/sebastian/data:/data -v $PWD:/pwd -w /pwd --memory 100gb -i -t tf_docker phire-eval

