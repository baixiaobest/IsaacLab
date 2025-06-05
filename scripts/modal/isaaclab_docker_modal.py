import os
import modal

# Use NVIDIA's official Isaac Sim image
gpu_image = modal.Image.from_registry(
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
    # Install additional system dependencies if needed
    "apt-get update && apt-get install -y git wget",
    "apt-get update && apt-get install -y curl wget git",
    "yes | apt install cmake build-essential",
    # Install X11 libraries - keep these
    "apt-get install -y libglu1-mesa libgl1-mesa-glx libxinerama1 libxcursor1 libxi6 libxrandr2",
    "apt-get install -y libsm6 libxt6 libxrender1 libice6 zenity xvfb",
).add_local_dir('.', remote_path='/root/IsaacLab', 
                ignore=['.github', '_isaac_sim', '.vscode',
                        'logs', 'docker', 'outputs', 'docs', 
                        '.aws', '.git', '__pycache__'],
                copy=True
).run_commands("yes | sh /root/IsaacLab/isaaclab.sh --install"
).env({
    "DISPLAY": ":99", 
    "DEBIAN_FRONTEND": "noninteractive", 
    "ENABLE_CAMERAS": "1", 
    "REMOTE_DEPLOYMENT": "1",
    "PYTHONPATH": "/root/IsaacLab/source/:/root/IsaacLab/source/isaaclab/",
    "ACCEPT_EULA": "Y",
    # Switch from Vulkan to EGL
    "CARB_GRAPHICS_API": "opengl",
    "CARB_USE_VULKAN_RENDERER": "0",
    "ISAAC_SIM_HEADLESS": "1",
    # Additional EGL/OpenGL settings
    "MESA_GL_VERSION_OVERRIDE": "4.6",
    "MESA_GLSL_VERSION_OVERRIDE": "460"
})

# Create app with the image
app = modal.App("isaaclab-training", image=gpu_image)

# Define persistent volume for data storage
isaaclab_logs_volume = modal.Volume.from_name("isaaclab_logs")
dump_volume = modal.Volume.from_name("dump")

@app.function(
    gpu='T4',
    timeout=int(4*3600),   # Increase timeout for GPU training
    volumes={"/root/IsaacLab/logs": isaaclab_logs_volume, 
             "/root/.local/share/ov/data/Kit/Isaac-Sim/4.5/": dump_volume},
)
def train_isaaclab():
    import os
    import time

    os.system("mkdir -p ~/.local/share/ov/data/Kit/Isaac-Sim/4.5/")
    os.system("chmod -R 777 ~/.local/share/ov")

    # Check GPU availability
    os.system("nvidia-smi")
    
    # Start virtual display with GPU acceleration
    os.system("Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &")
    time.sleep(3)
    
    # Set runtime GPU environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    os.chdir("/root/IsaacLab")

    # GPU-accelerated training command
    training_command = (
        f"cd /root/IsaacLab && "
        f"yes Yes | ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py "
        f"--task Isaac-Velocity-Rough-Unitree-Go2-Teacher-v0 "
        f"--num_envs 2048 "     # Reasonable for A10G GPU
        f"--headless "
        f"--video "
        f"--max_iterations 1500 "
        f"--sim_device cuda:0 "  # Use GPU for simulation
        f"--rl_device cuda:0"    # Use GPU for RL training
    )
    
    # Execute the training command
    print(f"Running training command: {training_command}")

    try:
        result = os.system(training_command)

        if result != 0:
            # Always copy crash dumps to persistent volume before container terminates
            print("Copying crash dumps to persistent volume...")
            os.system("cp -r ~/.local/share/ov/data/Kit/Isaac-Sim/4.5/*.zip /root/IsaacLab/logs/crash_dumps/")
            os.system("cp -r ~/.local/share/ov/data/Kit/Isaac-Sim/4.5/*.txt /root/IsaacLab/logs/crash_dumps/")
            return f"Training failed with exit code: {result}"
    
    finally:
        # Always copy crash dumps to persistent volume before container terminates
        print("Copying crash dumps to persistent volume...")
        os.system("cp -r ~/.local/share/ov/data/Kit/Isaac-Sim/4.5/*.zip /root/IsaacLab/logs/crash_dumps/")
        os.system("cp -r ~/.local/share/ov/data/Kit/Isaac-Sim/4.5/*.txt /root/IsaacLab/logs/crash_dumps/")
        
        # List the crash dumps that were saved
        os.system("ls -la /root/IsaacLab/logs/crash_dumps/")
    
    return "Training complete. Results saved to /root/IsaacLab/logs."

@app.local_entrypoint()
def main():
    result = train_isaaclab.remote()
    print(result)