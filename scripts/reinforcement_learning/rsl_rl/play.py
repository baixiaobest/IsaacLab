# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

def _is_interactive_backend():
    """Check if matplotlib is using an interactive backend.
    
    Returns:
        bool: True if the backend is interactive (can show windows), False otherwise.
    """
    actual_backend = matplotlib.get_backend().lower()
    return actual_backend not in ['agg', 'pdf', 'ps', 'svg', 'cairo']

# PLACEHOLDER: Extension template (do not remove this comment)

# Visualization update frequency
PLOT_UPDATE_BATCH_SIZE = 10  # Update plots every N updates


class VisualizationTracker:
    """Tracks and visualizes episode termination and metrics distributions."""
    
    def __init__(self, batch_size=PLOT_UPDATE_BATCH_SIZE, output_dir=None):
        """Initialize visualization tracking components.
        
        Args:
            batch_size: Number of updates to batch before redrawing plots.
            output_dir: Directory to save plots if display is not available. If None, uses current directory.
        """
        self.batch_size = batch_size
        self.output_dir = output_dir if output_dir else os.getcwd()
        # Check the actual backend being used
        if not _is_interactive_backend():
            if not hasattr(self, '_display_warning_printed'):
                print("[INFO] Using non-interactive backend. Plots will be saved to files in:", self.output_dir)
                self._display_warning_printed = True
        
        # Termination tracking
        self.termination_counts = {}
        self.termination_update_counter = 0
        
        # Metrics tracking
        self.metrics_history = {}
        self.metrics_update_counter = 0
        self.metrics_axes = {}
        self.fig_metrics = None
        self._metrics_plot_saved = False
        
        # Initialize termination plot
        self.fig_term, self.ax_term = plt.subplots(figsize=(10, 8))
        
        # Check if backend is actually interactive before trying to show
        is_interactive_backend = _is_interactive_backend()
        
        if is_interactive_backend:
            try:
                self.fig_term.canvas.manager.set_window_title('Episode Terminations')
                plt.show(block=False)
            except Exception:
                print("[INFO] Could not show plot window. Plots will be saved to files in:", self.output_dir)
    
    def track_and_update(self, extras):
        """Track termination and metrics data from environment step and update plots.
        
        Args:
            extras: Dictionary containing log data from environment step.
        """
        termination_needs_update = False
        metrics_needs_update = False
        metrics_discovered = False
        goal_reached = False
        # Check if any terminations occurred in this step
        for key in extras['log'].keys():
            if key.startswith('Episode_Termination'):
                value = extras['log'][key]
                if value > 0:  # Only track when terminations occur
                    if key not in self.termination_counts:
                        self.termination_counts[key] = 0
                    if key == 'Episode_Termination/goal_reached':
                        goal_reached = True
                    self.termination_counts[key] += int(value)
                    termination_needs_update = True
        
        # Track all metrics (they're all updated at termination)
        for key in extras['log'].keys():
            if key.startswith('Metrics/'):
                value = extras['log'][key]
                # Skip success-only metrics if goal was not reached
                if not goal_reached:
                    continue
                # Initialize tracking for new metrics
                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                    metrics_discovered = True
                # Add value to history
                if isinstance(value, (int, float)):
                    self.metrics_history[key].append(float(value))
                    metrics_needs_update = True
        
        # Increment counter and update termination plot only when batch size is reached
        if termination_needs_update:
            self.termination_update_counter += 1
            if self.termination_update_counter >= self.batch_size:
                self._update_termination_plot()
                self.termination_update_counter = 0
        
        # Create metrics figure if new metrics were discovered
        if metrics_discovered and self.fig_metrics is None:
            self._create_metrics_figure()
        
        # Increment counter and update metrics plots only when batch size is reached
        if metrics_needs_update:
            self.metrics_update_counter += 1
            if self.metrics_update_counter >= self.batch_size and self.fig_metrics is not None:
                self._update_metrics_plots()
                self.metrics_update_counter = 0
    
    def _update_termination_plot(self):
        """Update the termination distribution plot with current counts."""
        if not self.termination_counts:
            return
        
        self.ax_term.clear()
        self.ax_term.axis('on')
        keys = list(self.termination_counts.keys())
        values = list(self.termination_counts.values())
        
        # Clean up labels for better readability
        clean_labels = [k.replace('Episode_Termination/', '').replace('_', ' ') for k in keys]
        
        # Create labels with counts
        labels_with_counts = [f"{label}\n(n={value})" for label, value in zip(clean_labels, values)]
        
        # Create pie chart
        wedges, texts, autotexts = self.ax_term.pie(
            values,
            labels=labels_with_counts,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10},
            colors=plt.cm.Set3.colors
        )
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        self.ax_term.set_title('Episode Termination Distribution', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Check if backend is actually interactive
        is_interactive_backend = _is_interactive_backend()
        
        if is_interactive_backend:
            try:
                self.fig_term.canvas.draw()
                self.fig_term.canvas.flush_events()
            except Exception:
                # Fallback to saving if display fails
                is_interactive_backend = False
        
        # Always save if backend is non-interactive
        if not is_interactive_backend:
            # Save plot to file
            os.makedirs(self.output_dir, exist_ok=True)
            save_path = os.path.join(self.output_dir, 'episode_terminations.png')
            self.fig_term.savefig(save_path, dpi=150, bbox_inches='tight')
    
    def _update_metrics_plots(self):
        """Update the metrics distribution plots."""
        if not self.metrics_history or self.fig_metrics is None:
            return
        for metric_name, values in self.metrics_history.items():
            if metric_name not in self.metrics_axes or len(values) == 0:
                continue
            
            ax = self.metrics_axes[metric_name]
            ax.clear()
            
            # Create histogram
            ax.hist(values, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
            
            # Add vertical dashed line for mean
            mean_value = np.mean(values)
            ax.axvline(mean_value, color='red', linestyle='--', linewidth=2.5, label=f'Mean: {mean_value:.3f}')
            
            ax.set_xlabel('Value', fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            
            # Clean up metric name for title
            clean_name = metric_name.replace('Metrics/', '').replace('_', ' ').title()
            ax.set_title(f'{clean_name}\n(n={len(values)}, mean={mean_value:.3f})', 
                        fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        
        # Check if backend is actually interactive
        is_interactive_backend = _is_interactive_backend()
        
        if is_interactive_backend:
            try:
                self.fig_metrics.canvas.draw()
                self.fig_metrics.canvas.flush_events()
            except Exception:
                # Fallback to saving if display fails
                is_interactive_backend = False
        
        # Always save if backend is non-interactive
        if not is_interactive_backend:
            # Save plot to file
            os.makedirs(self.output_dir, exist_ok=True)
            save_path = os.path.join(self.output_dir, 'metrics_distributions.png')
            self.fig_metrics.savefig(save_path, dpi=150, bbox_inches='tight')
            if not self._metrics_plot_saved:
                print(f"[INFO] Saved metrics plots to: {save_path}")
                self._metrics_plot_saved = True
    
    def _create_metrics_figure(self):
        """Create figure and axes for metrics distributions."""
        num_metrics = len(self.metrics_history)
        cols = min(3, num_metrics)
        rows = (num_metrics + cols - 1) // cols
        self.fig_metrics, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        
        # Check if backend is actually interactive before trying to show
        is_interactive_backend = _is_interactive_backend()
        
        if is_interactive_backend:
            try:
                self.fig_metrics.canvas.manager.set_window_title('Metrics Distributions')
                plt.show(block=False)
            except Exception:
                pass
        
        # Flatten axes array for easier indexing
        if num_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if num_metrics > 1 else [axes]
        
        # Map metric names to axes
        for idx, metric_name in enumerate(sorted(self.metrics_history.keys())):
            self.metrics_axes[metric_name] = axes[idx]
        
        # Hide unused subplots
        for idx in range(num_metrics, len(axes)):
            axes[idx].set_visible(False)
    
    def close(self):
        """Close all visualization windows and save final plots if in headless mode."""
        # Check if backend is actually interactive
        is_interactive_backend = _is_interactive_backend()
        
        # Save final plots if backend is non-interactive
        if not is_interactive_backend:
            if self.termination_counts:
                self._update_termination_plot()
            if self.metrics_history and self.fig_metrics is not None:
                self._update_metrics_plots()
        plt.close('all')


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # Initialize visualization tracker
    # Save plots to the same directory as checkpoints if display is not available
    plot_output_dir = os.path.join(log_dir, "plots")
    viz_tracker = VisualizationTracker(output_dir=plot_output_dir)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, reward, dones, extras = env.step(actions)

        # Track and visualize episode termination and metrics distributions
        viz_tracker.track_and_update(extras)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    viz_tracker.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
