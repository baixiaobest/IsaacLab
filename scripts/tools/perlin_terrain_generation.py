import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2

def custom_perlin_noise(width, length, scale, amplitudes, lacunarity=2.0, seed=None):
    from noise import pnoise2
    noise_grid = np.zeros((length, width))
    max_val = 0  # for normalization
    for i in range(length):
        for j in range(width):
            x = j / scale
            y = i / scale
            value = 0
            frequency = 1.0
            for amp in amplitudes:
                value += amp * pnoise2(x * frequency, y * frequency, repeatx=width, repeaty=length, base=seed or 0)
                frequency *= lacunarity
                max_val += amp
            noise_grid[i][j] = value
    # Normalize to [0, 1]
    noise_grid = (noise_grid - noise_grid.min()) / (noise_grid.max() - noise_grid.min())
    return noise_grid

def generate_perlin_noise(width, length, scale=10.0, octaves=1, persistence=0.5, lacunarity=2.0, seed=None):
    noise_grid = np.zeros((length, width))
    for i in range(length):
        for j in range(width):
            x = j / scale
            y = i / scale
            noise_val = pnoise2(x, y, octaves=octaves, persistence=persistence,
                                lacunarity=lacunarity, repeatx=width, repeaty=length, base=seed or 0)
            noise_grid[i][j] = noise_val
    noise_grid = (noise_grid - noise_grid.min()) / (noise_grid.max() - noise_grid.min())
    return noise_grid

def stepped_interp(x, steps):
    """Convert values in [-1, 1] to stepped levels with given number of steps in [-1, 1]."""
    if steps < 2:
        raise ValueError("steps must be ≥ 2")
    # Map from [-1, 1] → [0, 1]
    x_norm = (x + 1) / 2
    # Quantize into steps
    x_stepped = np.floor(x_norm * steps) / (steps - 1)
    # Map back to [-1, 1]
    return x_stepped * 2 - 1

def generate_all_noise_maps(generator_func, parameter_sets, seed=None, step_levels=None):
    """Generate and optionally quantize all noise maps."""
    noise_maps = []
    for params in parameter_sets:
        try:
            noise = generator_func(**params, seed=seed)
            if step_levels is not None:
                noise = stepped_interp(noise, step_levels)
        except Exception as e:
            noise = None
        noise_maps.append(noise)
    return noise_maps

def threshold_noise_maps(noise, threshold):
    """Apply a threshold to the noise maps."""
    return np.where(noise < threshold, 0, noise)

def indices_of_nonzero(noise: np.ndarray) -> np.ndarray:
    """Get indices of non-zero elements in the noise map."""
    return np.argwhere(noise > 0)

# -------------------------------------
# 2. Plotting Functions
# -------------------------------------
def plot_2d_heatmaps(noise_maps, parameter_sets, title="Perlin Noise 2D Heatmap with Colorbar", threshold=None):
    fig, axes = plt.subplots(1, 2, figsize=(25, 15))
    for ax, noise, params in zip(axes.flat, noise_maps, parameter_sets):
        if noise is not None:
            if threshold is not None:
                noise = threshold_noise_maps(noise, threshold)
            im = ax.imshow(noise, cmap='terrain', origin='lower')
            ax.set_title(f"Oct: {params.get('octaves', 'N/A')} | Pers: {params.get('persistence', 'N/A')} | Lac: {params.get('lacunarity', 'N/A')}")
            fig.colorbar(im, ax=ax, shrink=0.8, label='Normalized Height')
            ax.axis('off')
        else:
            ax.set_title("Error")
            ax.text(0.5, 0.5, "Generation failed", ha='center')
            ax.axis('off')
    plt.suptitle(title, fontsize=16)
    # plt.tight_layout()

def plot_3d_surfaces(noise_maps, parameter_sets, title="3D Terrain Visualization with Colorbars"):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(25, 15))
    for i, (noise, params) in enumerate(zip(noise_maps, parameter_sets)):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        if noise is not None:
            x = np.linspace(0, 1, noise.shape[1])
            y = np.linspace(0, 1, noise.shape[0])
            x, y = np.meshgrid(x, y)
            surf = ax.plot_surface(x, y, noise, cmap='terrain', linewidth=0, antialiased=True)
            ax.set_title(f"Oct: {params.get('octaves', 'N/A')} | Pers: {params.get('persistence', 'N/A')} | Lac: {params.get('lacunarity', 'N/A')}")
            ax.set_axis_off()
            ax.set_zlim(-1, 1)
            fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, aspect=10, label='Normalized Height')
        else:
            ax.set_title("Error")
            ax.axis('off')
    plt.suptitle(title, fontsize=18)
    # plt.tight_layout()


if __name__ == "__main__":
    parameter_sets_1 = [
        {"width": 1000, "length": 1000, "scale": 200, "octaves": 1, "persistence": 0.5, "lacunarity": 2.0},
        {"width": 500, "length": 500, "scale": 100, "octaves": 1, "persistence": 0.5, "lacunarity": 2.0},
    ]

    parameter_sets_2 = [
        {"width": 1000, "length": 1000, "scale": 1000, "amplitudes": [0.4, 1.0, 0.0, 0.1, 0.0, 0.0, 0.01, 0.0, 0.002, 0.0, 0.0005], "lacunarity": 2.0},
        {"width": 1000, "length": 1000, "scale": 1000, "amplitudes": [0.4, 1.0, 0.2, 0.1, 0.0, 0.0, 0.01, 0.0, 0.002, 0.0, 0.0005], "lacunarity": 2.0},

    ]

    parameter_sets_3 = [
        {"width": 1000, "length": 1000, "scale": 20, "amplitudes": [0.5, 0.3, 0.5, 1.0], "lacunarity": 2.0},
        {"width": 1000, "length": 1000, "scale": 20, "amplitudes": [0.5, 0.3, 0.5, 1.0, 1.0], "lacunarity": 2.0}
    ]

    selected_parameter_set = parameter_sets_3
    selected_generator = custom_perlin_noise

    noise_maps = generate_all_noise_maps(selected_generator, selected_parameter_set, seed=57)

    # Plot 2D and 3D
    plot_2d_heatmaps(noise_maps, selected_parameter_set, threshold=0.83)
    plot_3d_surfaces(noise_maps, selected_parameter_set)
    plt.show()
