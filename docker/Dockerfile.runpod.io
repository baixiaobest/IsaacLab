# Use the NVIDIA Isaac Sim 4.5.0 base image
FROM nvcr.io/nvidia/isaac-sim:4.5.0

# Set EULA acceptance environment variable
ENV ACCEPT_EULA=Y

# Install git (not included in the base image)
RUN apt-get update && apt-get install -y git && apt-get clean

# Install build essential
run yes | apt-get install cmake build-essential

RUN yes | apt-get install pip

# Clone the repository and initialize submodules
WORKDIR /code
RUN git clone --recurse-submodules https://github.com/baixiaobest/IsaacLab.git \
    && cd IsaacLab \
    && git submodule update --init --recursive

RUN git clone https://github.com/baixiaobest/occupancy_prediction.git

# Set the working directory to IsaacLab and make the script executable
WORKDIR /code/IsaacLab

# link the isaac sim
RUN ln -s /isaac-sim _isaac_sim

RUN chmod +x isaaclab.sh

ENV TERM=xterm

RUN ./isaaclab.sh -p -m pip install numpy==1.26.0

# Run the installation script with the -i flag (install mode)
RUN bash ./isaaclab.sh -i

RUN ./isaaclab.sh -p -m pip install wandb==0.23.1

RUN ./isaaclab.sh -p -m pip install noise

# As previous install will update numpy back to the newest
RUN ./isaaclab.sh -p -m pip install numpy==1.26.0

# Setup for runpod.io
ENTRYPOINT [""]
CMD ["sleep", "infinity"]