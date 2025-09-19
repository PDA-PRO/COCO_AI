FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /coco
COPY reference ./reference
COPY desc ./desc
ADD wpc_bundle.pt ./wpc_bundle.pt
ADD main.py ./main.py
ADD task_detail.json ./task_detail.json
RUN python -m zipfile -e desc/desc.zip desc

COPY requirements.txt /coco/requirements.txt
RUN pip install -r /coco/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--port", "8000", "--host", "0.0.0.0"]