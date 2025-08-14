FROM docker.1ms.run/pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

ENV PYTHONUNBUFFERED 1
ENV http_proxy="http://172.16.40.42:7890"
ENV https_proxy="http://172.16.40.42:7890"
ENV HTTP_PROXY="http://172.16.40.42:7890"
ENV HTTPS_PROXY="http://172.16.40.42:7890"

WORKDIR /usr/src/app

# Install packages
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --resume-retries 3 -r requirements.txt
RUN pip install --no-cache-dir "httpx[socks]"
# RUN pip install --no-cache-dir nvidia-cudnn-cu12
RUN pip uninstall -y nvidia-cudnn-cu12
RUN conda install -c nvidia cudnn

# If you want to use Melo TTS, you also need to run:
# RUN python -m unidic download
# 破天朝网络，服务器上下载超时，只能下载好了copy进去
COPY unidic-dic-complete /opt/conda/lib/python3.11/site-packages/unidic/dicdir

COPY . .
