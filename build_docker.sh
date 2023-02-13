#!/bin/sh

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

docker build $SCRIPTPATH -t tf_docker
