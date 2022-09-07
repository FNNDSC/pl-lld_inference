# Docker file for lld_inference ChRIS plugin app
#
# Build with
#
#   docker build -t <name> .
#
# For example if building a local version, you could do:
#
#   docker build -t local/pl-lld_inference .
#
# In the case of a proxy (located at 192.168.13.14:3128), do:
#
#    docker build --build-arg http_proxy=http://192.168.13.14:3128 --build-arg UID=$UID -t local/pl-lld_inference .
#
# To run an interactive shell inside this container, do:
#
#   docker run -ti --entrypoint /bin/bash local/pl-lld_inference
#
# To pass an env var HOST_IP to container, do:
#
#   docker run -ti -e HOST_IP=$(ip route | grep -v docker | awk '{if(NF==11) print $9}') --entrypoint /bin/bash local/pl-lld_inference
#
FROM alpine:latest as download

WORKDIR /tmp
ADD https://fnndsc.childrens.harvard.edu/LLD/weights/model.tar.gz /tmp/model.tar.gz
RUN ["tar", "xf", "model.tar.gz"]


FROM tensorflow/tensorflow:latest-gpu-py3
LABEL maintainer="FNNDSC <dev@babyMRI.org>"

# Install `wget` to install `miniconda`
RUN apt-get install wget

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.1-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -u -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# install the cuda version required to work with tensorflow
# chcek here: https://www.tensorflow.org/install/source#gpu to find out the right version
RUN conda install -c conda-forge cudatoolkit=10.0

COPY --from=download /tmp/model /usr/local/lib/lld

WORKDIR /usr/local/src

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install .

CMD ["lld_inference", "--help"]
