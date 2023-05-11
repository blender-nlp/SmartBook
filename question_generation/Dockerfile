FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

# Installing the required packages
RUN apt update -y && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt install -y git \
    curl \
    python3.8  \
    python3.8-distutils \
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
    python3.8 get-pip.py

RUN mkdir /data/

WORKDIR /data

RUN git clone https://huggingface.co/navteca/quora-roberta-base
RUN git clone https://huggingface.co/castorini/t5-base-canard

WORKDIR /code
ADD . / /code/

RUN pip install -r requirements.txt

RUN python3.8 -m nltk.downloader punkt

ENV CUDA_VISIBLE_DEVICES=0

# Set Environment variables
ENV INPUT_DIR /var/spool/input/
ENV OUTPUT_DIR /var/spool/output/
ENV OPENAI_KEY ""

CMD python3.8 /code/run.py --openai_api_key "${OPENAI_KEY}" --expand_question_model_path /data/t5-base-canard/ --duplicate_question_model_path /data/quora-roberta-base  --input_dir "${INPUT_DIR}" --output_dir "${OUTPUT_DIR}"