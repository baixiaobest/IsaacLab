# depth_logger.py
import os
import torch
from isaaclab.managers.recorder_manager import RecorderTerm

class DepthLogger(RecorderTerm):
    def record_post_step(self):
        # lazy-init a frame counter and save directory
        if not hasattr(self, "frame"):
            self.frame = 0
            self.save_dir = "/home/azureuser/Desktop/dataset/depth_height"
            os.makedirs(self.save_dir, exist_ok=True)

        # pull tensors off the sim and move to CPU
        depth_t  = self._env.scene.sensors["depth_sensor"].data.output["distance_to_image_plane"]
        height_t = self._env.scene.sensors["height_scanner"].data.ray_hits_w

        # squeeze any singleton dims, then CPU-ize
        depth_t  = depth_t.squeeze(-1).cpu()    # [B, H, W]
        height_t = height_t.cpu()               # [B, N, ...]

        batch_size = depth_t.shape[0]
        for env_id in range(batch_size):
            payload = {
                "depth": depth_t[env_id],
                "height": height_t[env_id],
            }
            fn = f"env{env_id:02d}_frame{self.frame:06d}.pt"
            # torch.save(payload, os.path.join(self.save_dir, fn))

        self.frame += 1

        return "sensors", {
            "depth": depth_t,
            "height_scanner": height_t
        }