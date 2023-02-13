FROM tensorflow/tensorflow:1.15.5-gpu

RUN apt-get install -y gcc gfortran libfftw3-dev libblas-dev liblapack-dev

WORKDIR /PhIRE

COPY docker_requirements.txt docker_requirements.txt
RUN pip install -r docker_requirements.txt

COPY python/ python/
COPY setup.py setup.py

RUN pip install -e .
