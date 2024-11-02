FROM tensorflow/tensorflow:latest-gpu

ARG USER
RUN useradd $USER

COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt

ENTRYPOINT []
