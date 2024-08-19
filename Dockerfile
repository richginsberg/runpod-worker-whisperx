# Base image -> https://github.com/runpod/containers/blob/main/official-templates/base/Dockerfile
# DockerHub -> https://hub.docker.com/r/runpod/base/tags
FROM runpod/base:0.4.0-cuda11.8.0

# Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b && \
    rm -f miniconda.sh

RUN apt update && apt install -y ffmpeg

COPY src/environment.yaml .
RUN conda env create -f environment.yaml

# Trigger WhisperX to download necessary models, so we can embed them in the image
ARG hftoken
COPY src/audio_en.mp3 .
RUN /bin/bash -c "source activate whisperx; whisperx --hf_token $hftoken --model large-v3 --diarize --compute_type float32 --lang en ./audio_en.mp3"
ADD src .
COPY src .

#CMD python3.11 -u /handler.py
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "whisperx", "python3.11", "-u", "/handler.py"]