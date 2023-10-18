FROM python:3.10.13-slim

WORKDIR /faiss
RUN apt-get update && apt-get install git -y

EXPOSE 8000

CMD tail -f /dev/null