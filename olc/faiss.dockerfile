FROM python:3.10.13-slim

WORKDIR /faiss

COPY ./faiss/main.py /faiss/main.py
COPY requirements.txt /faiss/requirements.txt
RUN pip install -r /faiss/requirements.txt

EXPOSE 8000

CMD uvicorn main:app --port 8000 --host 0.0.0.0