FROM tensorflow/tensorflow:latest-gpu
LABEL author="kodamanbou0424@gmail.com"

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git

WORKDIR /workspace
EXPOSE 6006
