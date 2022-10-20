#!/bin/sh
docker login
docker build --tag themonocledhamster/kube-demo:latest .
docker push themonocledhamster/kube-demo:latest