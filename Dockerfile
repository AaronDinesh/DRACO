FROM nvcr.io/nvidia/jax:25.08-py3

ENV DEBIAN_FRONTEND=noninteractive

# 3. Install Python, pip, git, curl, and other essentials
RUN apt-get update && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y \
    python3.13 \
    python3.13-venv \
    python3-pip \
    git \
    curl \
    # Add any other system build dependencies your Python packages might need
    # e.g., build-essential if some packages compile C extensions
    build-essential \
    && \
    # Make python3.10 the default python3 (optional but good for consistency)
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 && \
    update-alternatives --set python3 /usr/bin/python3.13 && \
    # Clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 --version && python3 -m pip --version

# 4. Install uv
RUN echo "Installing uv..." && \
    curl -LsSf https://astral.sh/uv/install.sh | sh
# uv installs to $HOME/.local/bin. For root (default user in many base images), $HOME is /root.
# If base image has a non-root user, adjust $HOME or install uv to a system path.
ENV PATH="/root/.local/bin:${PATH}"
RUN uv --version

RUN mkdir ./DRACO

# 5. Copy project files and install dependencies using uv
COPY * ./DRACO

RUN echo "Creating Python 3.13.7 virtual environment with uv..." && \
    uv venv .venv --python python3.13.7 && \
    echo "Syncing dependencies into the virtual environment..." && \
    uv sync

CMD ["uv", "run", "python3", "main.py", "--input-maps=./data/Maps_ne_IllustrisTNG_SB28_z=0.00.npy", "--output-maps=./data/Maps_ne_IllustrisTNG_SB28_z=0.00.npy", "--cosmos-params=./data/params_SB28_IllustrisTNG.txt"]
