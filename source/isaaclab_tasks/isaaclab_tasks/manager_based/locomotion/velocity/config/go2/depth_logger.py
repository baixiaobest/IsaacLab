# depth_logger.py
import torch
from isaaclab.managers.recorder_manager import RecorderTerm

class DepthLogger(RecorderTerm):
    def record_post_step(self):
        # Debug print available sensors
        print("[DEBUG] Available sensors:", self._env.scene.sensors.keys())
        
        # Access depth camera data
        depth_sensor = self._env.scene.sensors["depth_sensor"]
        height_scanner = self._env.scene.sensors["height_scanner"]
        
        # Get depth data from the correct field (distance_to_image_plane)
        depth_data = depth_sensor.data.output["distance_to_image_plane"]
        
        # For RayCaster, we need to access ray_hits_w
        height_data = height_scanner.data.ray_hits_w
        
        # Debug print shapes
        print(f"[DEBUG] Depth data shape: {depth_data.shape}")
        print(f"[DEBUG] Height data shape: {height_data.shape}")
        
        # Ensure data is properly formatted for recording
        depth_data = depth_data.squeeze(-1)  # Remove last dimension if it's [batch, H, W, 1]
        
        # Return data in correct format for recorder
        return "sensors", {
            "depth": depth_data,
            "height_scanner": height_data
        }
