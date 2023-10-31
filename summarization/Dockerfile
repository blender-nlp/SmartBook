FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Installing the required packages
RUN apt update -y && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt install -y git \
    curl \
    python3.9  \
    python3.9-distutils \
    wget \
    nginx  \
    supervisor \
    nginx\
    gcc \
    g++

RUN wget https://github.com/git-lfs/git-lfs/releases/download/v2.9.0/git-lfs-linux-amd64-v2.9.0.tar.gz && \
    tar -xf git-lfs-linux-amd64-v2.9.0.tar.gz && \
    chmod 755 install.sh && \
    ./install.sh && \
    git lfs install && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.9 get-pip.py

WORKDIR /code
ADD . / /code/

RUN pip install -r requirements.txt

# Set Environment variables
ENV INPUT_DIR /var/spool/input/
ENV OUTPUT_DIR /var/spool/output/
ENV OPENAI_KEY ""
ENV GENERATION_MODEL_PATH ""

CMD python3.8 /code/run.py --openai_api_key "${OPENAI_KEY}" --input_dir "${INPUT_DIR}" --output_dir "${OUTPUT_DIR}" --generation_model_path "${GENERATION_MODEL_PATH}"