FROM jupyter/scipy-notebook

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY mnist/plot_digits_classification_end_sem.py ./plot_digits_classification_end_sem.py
COPY mnist/utils/utils.py ./utils/utils.py

ADD mnist/models ./models

USER root

WORKDIR "/home/jovyan/"

RUN chmod a+x models

RUN python plot_digits_classification_end_sem.py