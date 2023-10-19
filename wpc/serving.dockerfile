FROM python:3.10.13-slim

WORKDIR /coco
RUN apt-get update && apt-get install git -y && apt-get install gcc -y
COPY reference ./reference
COPY desc ./desc
ADD pytorch_model.bin ./pytorch_model.bin
ADD build.sh ./build.sh
ADD build.py ./build.py
ADD main.py ./main.py

COPY requirements.txt /coco/requirements.txt
RUN pip install -r /coco/requirements.txt
RUN ./build.sh

EXPOSE 8000

CMD uvicorn main:app --port 8000 --host 0.0.0.0