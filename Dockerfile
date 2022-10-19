FROM python:3.10-slim

WORKDIR /k8s-deployment
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY ./server   ./
CMD [ "python3", "-m" , "flask","--app","server", "run", "--host=0.0.0.0"]
