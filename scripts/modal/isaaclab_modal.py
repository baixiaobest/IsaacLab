import os
import modal

cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Define a container image with CUDA support
gpu_image = modal.Image.from_registry(
    # f"nvidia/cuda:{tag}", 
    "nvcr.io/nvidia/isaac-sim:4.5.0",
    add_python="3.10"
).pip_install(
    "requests",
    "tqdm",
    "toml",
    "flatdict",
    "pyyaml",
    "tensorboard",
    "wandb"
).run_commands(
    # Install system dependencies
    "apt-get update && apt-get install -y curl wget git",
    "yes | apt install cmake build-essential",
    # Install X11 libraries - keep these
    "apt-get install -y libglu1-mesa libgl1-mesa-glx libxinerama1 libxcursor1 libxi6 libxrandr2",
    "apt-get install -y libsm6 libxt6 libxrender1 libice6 zenity xvfb",
    # Install Vulkan - but remove specific driver packages
    "apt-get install -y vulkan-tools mesa-vulkan-drivers vulkan-validationlayers",
    # Python packages
    "pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121",
    "pip install --upgrade pip",
    "pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com",
    # Install isaac sim
    # "mkdir /root/",
    # "wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone%404.5.0-rc.36%2Brelease.19112.f59b3005.gl.linux-x86_64.release.zip -P /root",
    # "unzip /root/isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.linux-x86_64.release.zip -d /root/isaacsim/",
    # "rm /root/isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.linux-x86_64.release.zip"
).add_local_dir('.', remote_path='/root/IsaacLab', 
                ignore=['.github', '_isaac_sim', '.vscode', 'logs', 'docker', 'outputs', 'docs', '.aws', '.git'],
                copy=True
).run_commands("yes | sh /root/IsaacLab/isaaclab.sh --install"
).env({
    "DISPLAY": ":99", 
    "DEBIAN_FRONTEND": "noninteractive", 
    "ENABLE_CAMERAS": "1", 
    "REMOTE_DEPLOYMENT": "1",
    "PYTHONPATH": "/root/IsaacLab/source/:/root/IsaacLab/source/isaaclab/",
    "ACCEPT_EULA": "Y",
    # GPU-specific settings
    "OMNI_KIT_ACCEPT_EULA": "YES",
    # "NVIDIA_VISIBLE_DEVICES": "all",
    # "NVIDIA_DRIVER_CAPABILITIES": "1"
    })


# Create app with the image
app = modal.App("isaaclab-training", image=gpu_image)
dump_volume = modal.Volume.from_name("dump")

# Define persistent volume for data storage
isaaclab_logs_volume = modal.Volume.from_name("isaaclab_logs")

@app.function(
    gpu="L4",
    timeout=int(0.5*3600),  # Set timeout to 4 hours
    volumes={"/root/IsaacLab/logs": isaaclab_logs_volume,
             "/root/.local/share/ov/data/Kit/Isaac-Sim/4.5/": dump_volume},
)
def train_isaaclab():
    import os
    import time

    # Check GPU availability
    os.system("nvidia-smi")

    # Start virtual display properly
    os.system("Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX +render &")
    time.sleep(2)  # Give Xvfb time to initialize
    
    # Verify X server is running
    os.system("xdpyinfo -display :99 > /tmp/xdpyinfo.log 2>&1")

    os.chdir("/root/IsaacLab")

    # Step 3: Run training with the specified parameters
    training_command = (
        f"cd /root/IsaacLab && "
        # f"yes Yes | ./isaaclab.sh "
        f"ACCEPT_EULA=Y /isaac-sim/runheadless.sh"
        f"-p scripts/reinforcement_learning/rsl_rl/train.py "
        f"--task Isaac-Velocity-Rough-Unitree-Go2-Teacher-v0 "
        f"--num_envs 4096 "
        f"--headless "
        f"--max_iterations 1500"
    )
    
    # Execute the training command
    print(f"Running training command: {training_command}")
    os.system(training_command)
    
    return "Training complete. Results saved to /root/IsaacLab/logs."

@app.local_entrypoint()
def main():
    result = train_isaaclab.remote()
    print(result)