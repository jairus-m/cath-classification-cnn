# Use Python 3.12 as the base image
FROM python:3.12-slim

# Install uv & git
RUN pip install uv
RUN apt-get update && apt-get install -y git

# Expose the mlflow webserver port
EXPOSE 5000