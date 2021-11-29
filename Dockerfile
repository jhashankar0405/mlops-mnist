from ubuntu:20.04
COPY mnist mnist
COPY requirements.txt requirements.txt 
COPY api api

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install -r requirements.txt

WORKDIR /
CMD ["python3", "api/app.py"]
