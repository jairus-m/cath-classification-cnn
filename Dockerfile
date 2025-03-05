# Use Python 3.12 as the base image
FROM python:3.12-slim

# Install uv & git
RUN pip install uv
RUN apt-get update && apt-get install -y git


# Set up the working directory
WORKDIR /cath-cnn

# Copy your project files
COPY . /cath-cnn

# Create a virtual environment and install dependencies using uv
RUN uv venv /cath-cnn/.venv
ENV PATH="/cath-cnn/.venv/bin:$PATH"
RUN uv pip install --system -e ".[dev]"

# Expose the mlflow webserver port
EXPOSE 5000

# Keep container running indefinitely to attach VSCode via Remote Explorer Extension
CMD [ "sleep", "infinity" ]


### Build image
# In the CLI, run `docker build -t cath-cnn:latest .`

### Run container (make sure your machine/Docker Engine have enough memory)
# CLI: Run the image `docker run --memory=16g cath-cnn:latest` 
# GUI: Look for the image you just created and run a new container

### Use container as Dev Env for VSCode
# Use the VSCode Remote Explorer Extension to use the docker image as your dev env/IDE
