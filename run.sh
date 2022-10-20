#!/bin/sh
kubectl apply -k ./kube/base
kubectl port-forward deployments/deploy 5000:5000