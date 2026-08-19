"""Shared geometry constants for held temporal-lidar scans."""

# The held sensor fan and the world-aligned temporal grid intentionally use the
# same resolution.  Keeping this value here prevents the collector, scanner,
# and observation configuration from drifting apart.
TEMPORAL_LIDAR_RESOLUTION = 512
TEMPORAL_LIDAR_RAYS = TEMPORAL_LIDAR_RESOLUTION
TEMPORAL_LIDAR_NUM_BINS = TEMPORAL_LIDAR_RESOLUTION
