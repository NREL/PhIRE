FROM tensorflow/tensorflow:1.15.5-gpu
RUN apt-get install -y gcc gfortran libfftw3-dev libblas-dev liblapack-dev
RUN pip install matplotlib pyspharm pyshtools
WORKDIR /PhIRE
COPY python/ python/
COPY setup.py setup.py
RUN pip install -e .
