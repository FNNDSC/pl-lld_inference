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

# Using CPU optimized base image for Tensorflow v1.15
# https://www.intel.com/content/www/us/en/developer/articles/guide/optimization-for-tensorflow-installation-guide.html#docker_images
#
# LLDCode does not work with the official Tensorflow image, tensorflow/tensorflow:1.15.5-py3
# You get this error:
#
#    tensorflow.python.framework.errors_impl.UnimplementedError: The Conv2D op currently only supports the NHWC tensor format on the CPU.
#    The op was given the format: NCHW
#
# The image from Intel has a fix for the problem.
# Ref: https://github.com/onnx/onnx-tensorflow/issues/535

FROM gcr.io/deeplearning-platform-release/tf-cpu.1-15
LABEL maintainer="FNNDSC <dev@babyMRI.org>"

# download and unpack ML model weights
RUN curl https://fnndsc.childrens.harvard.edu/LLD/weights/model.tar.gz \
     | tar --transform 's/^model/lld/' -xvz -C /usr/local/lib

# install dependencies and helpful (?) tools
COPY requirements.txt .
RUN  apt-key adv --keyserver keyserver.ubuntu.com --recv A4B469963BF863CC && \
     apt update && apt -y install pciutils sudo kmod libgl1-mesa-glx ffmpeg libsm6 libxext6
RUN pip install --upgrade pip setuptools wheel
RUN pip install pyOpenSSL --upgrade
RUN pip install -r requirements.txt

COPY . .
RUN pip install .

CMD ["lld_inference", "--man"]
