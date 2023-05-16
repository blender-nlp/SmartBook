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

RUN git clone https://huggingface.co/gangiswag/tanda_roberta_large_asnq_wikiqa
RUN git clone https://huggingface.co/gangiswag/qa_nq

WORKDIR /code
ADD . / /code/

RUN pip3 install -r requirements.txt

RUN python3.8 -m spacy download en_core_web_sm

ENV CUDA_VISIBLE_DEVICES=0

# Set Environment variables
ENV INPUT_DIR /var/spool/input/
ENV OUTPUT_DIR /var/spool/output/
ENV SERPAPI_KEY ""
ENV START_DATE ""
ENV END_DATE ""

CMD python3.8 /code/run.py --serp_api_key "${SERPAPI_KEY}" --start_date "${START_DATE}" --end_date "${END_DATE}" --claim_model_path /data/qa_nq/ --verification_model_path /data/tanda_roberta_large_asnq_wikiqa --input_dir "${INPUT_DIR}" --output_dir "${OUTPUT_DIR}" --use_gpu