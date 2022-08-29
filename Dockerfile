FROM nvidia/cuda:11.7.0-runtime-ubuntu22.04
#FROM nvidia/cuda:11.2.1-cudnn8-runtime-ubuntu20.04
#FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
#FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"



RUN apt-get update  && \
    apt upgrade -y  && \
    apt install -y htop python3-dev wget
RUN apt install -y python3-pip python3-setuptools

RUN apt-get -y install build-essential
RUN pip3 install --upgrade setuptools

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh && mkdir root/.conda && sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b


RUN conda create -y -n ml python=3.9

COPY . src/
RUN /bin/bash -c "cd src \
    && source activate ml  \
    && pip install --upgrade cython \
    && pip install -r requirements.txt \
    && pip uninstall -y torch \
    && pip uninstall -y torchaudio \
    && pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116"

EXPOSE 8000
#ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
ENTRYPOINT /bin/bash -c "cd src \
    && source activate ml \
    && python -m uvicorn main:app --host 0.0.0.0 --port 8000"


