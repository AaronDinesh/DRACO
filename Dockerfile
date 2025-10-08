FROM nvcr.io/nvidia/jax:25.01-py3

ENV DEBIAN_FRONTEND=noninteractive

# 3. Install Python, pip, git, curl, and other essentials
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    git \
    curl \
    # Add any other system build dependencies your Python packages might need
    # e.g., build-essential if some packages compile C extensions
    build-essential \
    && \
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

# RUN mkdir /home/DRACO
# WORKDIR /home/DRACO



# COPY . ./

# RUN echo "Creating Python 3.13.7 virtual environment with uv..." && \
#     uv venv .venv --python python3.13.7 && \
#     uv sync
