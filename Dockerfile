FROM pytorchlightning/pytorch_lightning:base-cuda-py3.11-torch2.3-cuda12.1.1

# Set the working directory inside the container
WORKDIR /app

# Update package lists
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git wget ca-certificates \
      libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code to the container
# COPY . /app

# Set the command to run when the container starts
ENTRYPOINT ["python"]