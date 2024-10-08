# Use NVIDIA's CUDA image with a compatible Python version
FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu20.04

# Set environment variables to avoid prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and necessary dependencies
RUN apt-get update && \
    apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    portaudio19-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a symbolic link to make Python 3.10 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Set up a working directory for the project
WORKDIR /app

# Set up a virtual environment and activate it
RUN python3.10 -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Install pip dependencies
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio jupyter ipykernel && \
    pip install Cython packaging speechbrain transformers librosa pyaudio && \
    pip install git+https://github.com/NVIDIA/NeMo.git && \
    pip install scikit-learn essentia-tensorflow BeatNet && \
    pip install diffusers stable_diffusion_videos keybert

# Clone the Folly repository
RUN git clone https://github.com/Folly/Folly /app/Folly
# COPY . /app


# Set the working directory to the Folly project directory
WORKDIR /app/Folly

# Expose any ports that Folly might use (adjust if necessary)
EXPOSE 8888

# Set the command to run the demo.py script when the container starts
CMD ["python", "demo.py"]
