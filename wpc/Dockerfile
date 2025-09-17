FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /coco
RUN apt-get update && apt-get install -y gcc git unzip
COPY reference ./reference
COPY desc ./desc
ADD pytorch_model.bin ./pytorch_model.bin
ADD main.py ./main.py
ADD task_detail.json ./task_detail.json
RUN unzip desc/desc.zip -d desc

COPY requirements.txt /coco/requirements.txt
RUN pip install -r /coco/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--port", "8000", "--host", "0.0.0.0"]