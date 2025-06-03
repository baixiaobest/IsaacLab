# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates a rolling sphere with random initial velocity that resets every 10 seconds.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_rolling_sphere.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import torch
import random

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Demo of a rolling sphere that resets every 10 seconds.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext


def design_scene() -> RigidObject:
    """Designs the scene with a sphere rigid object."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.4,
            dynamic_friction=0.4,
            restitution=0.1
        )
        # Remove incorrect collision_props parameter
    )
    cfg.func("/World/defaultGroundPlane", cfg)
    
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    
    # Sphere
    sphere_cfg = RigidObjectCfg(
        prim_path="/World/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, 
                solver_velocity_iteration_count=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.3,
                dynamic_friction=0.3,
                restitution=0.5
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.6, 1.0), 
                metallic=0.5
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),  # Start a bit above the ground
            lin_vel=(0.0, 0.0, 0.0),  # Will be set randomly in reset_sphere
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )
    
    # Create the object
    sphere = RigidObject(cfg=sphere_cfg)
    
    return sphere


def reset_sphere(sphere: RigidObject, sim: SimulationContext):
    """Resets the sphere's position and gives it a random velocity."""
    # Reset position
    root_state = sphere.data.default_root_state.clone()
    sphere.write_root_pose_to_sim(root_state[:, :7])
    
    # Generate random velocity direction with fixed magnitude
    velocity_magnitude = 3.0
    theta = random.uniform(0, 2 * torch.pi)
    vx = velocity_magnitude * torch.cos(torch.tensor([theta]))
    vy = velocity_magnitude * torch.sin(torch.tensor([theta]))
    vz = 0.0  # No initial vertical velocity
    
    # Set random velocity
    velocity = torch.tensor([[vx.item(), vy.item(), vz]], device=sim.device)
    sphere.write_root_velocity_to_sim(
        torch.cat([velocity, torch.zeros_like(velocity)], dim=-1)  # [lin_vel, ang_vel]
    )
    
    # Reset internal state
    sphere.reset()
    
    print(f"[INFO]: Reset sphere with velocity: [{vx.item():.2f}, {vy.item():.2f}, {vz:.2f}]")


def run_simulator(sim: SimulationContext, sphere: RigidObject):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    reset_interval = 10.0  # Reset every 10 seconds of simulation time
    
    # Initial reset
    reset_sphere(sphere, sim)
    
    # Simulate physics
    while simulation_app.is_running():
        # Check if it's time to reset
        if sim_time >= reset_interval:
            # Reset timer
            sim_time = 0.0
            # Reset the sphere
            reset_sphere(sphere, sim)
            
        # Perform step
        sim.step()
        
        # Update sim-time
        sim_time += sim_dt
        
        # Update buffers
        sphere.update(sim_dt)
        
        # Print sphere position and velocity occasionally
        if int(sim_time / sim_dt) % 100 == 0:
            pos = sphere.data.root_pos_w[0].cpu().numpy()
            vel = sphere.data.root_lin_vel_w[0].cpu().numpy()
            print(f"Time: {sim_time:.2f}s, Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], "
                  f"Velocity: [{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}]")


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    # Set main camera
    sim.set_camera_view(eye=[5.0, 5.0, 3.0], target=[0.0, 0.0, 0.0])
    
    # Design scene
    sphere = design_scene()
    
    # Play the simulator
    sim.reset()
    
    # Now we are ready!
    print("[INFO]: Setup complete...")
    
    # Run the simulator
    run_simulator(sim, sphere)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
