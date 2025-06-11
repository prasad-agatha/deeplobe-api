FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=TRUE

RUN apt-get update 

RUN apt-get install -y python3-pip python3-dev libgl1-mesa-glx locales locales-all libglib2.0-0 libsm6 libxrender1 libxext6\
    && cd /usr/local/bin \
    && ln -s /usr/bin/python3 python \
    && pip3 install --upgrade pip

WORKDIR /app

COPY /requirements ./requirements

COPY requirements.txt ./

RUN pip3 install -r requirements.txt

COPY . .

RUN python manage.py collectstatic --noinput