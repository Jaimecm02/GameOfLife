import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import convolve2d
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class LeniaConfig:
    """Configuration settings for Lenia."""
    size: int
    initial_live_probability: float
    time_step: float = 0.05       # Slower updates for stability
    kernel_radius: int = 13
    growth_center: float = 0.135  # Adjusted for better stability
    growth_width: float = 0.025   # Wider growth range
    kernel_rings: int = 4
    save_frames: bool = False
    output_dir: str = 'lenia_output'
    color_scheme: str = 'plasma'  # Added color scheme parameter
    blur_factor: float = 0.2      # For motion trails

class Lenia:
    def __init__(self, config: LeniaConfig):
        self.config = config
        self.grid = self._initialize_grid()
        self.kernel = self._create_kernel()
        self.frame_count = 0
        self.trail_buffer = np.zeros_like(self.grid)  # Added for motion trails
        
    def _initialize_grid(self) -> np.ndarray:
        """Initialize the grid with a stable pattern."""
        grid = np.zeros((self.config.size, self.config.size))
        center = self.config.size // 2
        pattern_size = self.config.size // 5
        
        # Create a more dense central pattern
        for _ in range(pattern_size * 2):
            x = center + np.random.randint(-pattern_size//2, pattern_size//2)
            y = center + np.random.randint(-pattern_size//2, pattern_size//2)
            if 0 <= x < self.config.size and 0 <= y < self.config.size:
                grid[y, x] = np.random.random() * 0.5 + 0.5
                
                # Add some neighboring cells for stability
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.config.size and 0 <= ny < self.config.size:
                        grid[ny, nx] = np.random.random() * 0.3 + 0.3
        
        return grid.astype(np.float64)
    
    def _create_spiral_pattern(self) -> np.ndarray:
        """Create a spiral initial pattern."""
        grid = np.zeros((self.config.size, self.config.size))
        center = self.config.size // 2
        max_radius = min(center // 2, 40)
        
        for r in range(max_radius):
            angle = r * 0.5
            x = center + int(r * np.cos(angle))
            y = center + int(r * np.sin(angle))
            if 0 <= x < self.config.size and 0 <= y < self.config.size:
                grid[y, x] = 1
        
        return grid
    
    def _create_cross_pattern(self) -> np.ndarray:
        """Create a cross-shaped initial pattern."""
        grid = np.zeros((self.config.size, self.config.size))
        center = self.config.size // 2
        size = self.config.size // 8
        
        grid[center-size:center+size, center] = 1
        grid[center, center-size:center+size] = 1
        return grid
    
    def _create_random_clusters(self) -> np.ndarray:
        """Create random clusters of cells."""
        grid = np.zeros((self.config.size, self.config.size))
        num_clusters = np.random.randint(3, 7)
        
        for _ in range(num_clusters):
            x = np.random.randint(self.config.size)
            y = np.random.randint(self.config.size)
            radius = np.random.randint(5, 15)
            
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    if dx*dx + dy*dy <= radius*radius:
                        nx, ny = (x + dx) % self.config.size, (y + dy) % self.config.size
                        grid[ny, nx] = np.random.random() * 0.8 + 0.2
        
        return grid
    
    def _create_symmetrical_pattern(self) -> np.ndarray:
        """Create a symmetrical pattern."""
        grid = np.zeros((self.config.size, self.config.size))
        center = self.config.size // 2
        pattern_size = self.config.size // 6
        
        # Create one quarter of the pattern
        quarter = np.random.random((pattern_size, pattern_size)) < 0.3
        
        # Mirror horizontally and vertically
        half = np.hstack((quarter, np.fliplr(quarter)))
        full = np.vstack((half, np.flipud(half)))
        
        # Place in center of grid
        start = center - pattern_size
        grid[start:start+2*pattern_size, start:start+2*pattern_size] = full
        return grid
    
    def _create_kernel(self) -> np.ndarray:
        """Create a more complex kernel with varying ring weights."""
        radius = self.config.kernel_radius
        L = np.arange(-radius, radius + 1)
        X, Y = np.meshgrid(L, L)
        R = np.sqrt(X**2 + Y**2) / radius
        
        kernel = np.zeros_like(R)
        rings = self.config.kernel_rings
        
        # Create rings with varying weights
        for i in range(rings):
            ring_center = (i + 1) / rings
            ring_width = 1.0 / (rings * 3)
            weight = 1.0 - (i / rings) * 0.3  # Outer rings have less influence
            kernel += weight * np.exp(-((R - ring_center) ** 2) / (2 * ring_width ** 2))
        
        kernel[R > 1] = 0
        return kernel / np.sum(kernel)
    
    def _growth_function(self, x: np.ndarray) -> np.ndarray:
        """Calculate growth based on neighborhood sum using a bell curve."""
        mu = self.config.growth_center
        sigma = self.config.growth_width
        return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    def update(self) -> np.ndarray:
        """Update the grid with motion trails."""
        # Calculate neighborhood influence
        convolved = convolve2d(self.grid, self.kernel, mode='same', boundary='wrap')
        
        # Apply growth function
        growth_rate = self._growth_function(convolved)
        
        # Update grid with time step
        new_grid = self.grid + self.config.time_step * (2 * growth_rate - 1)
        new_grid = np.clip(new_grid, 0, 1)
        
        # Update trail buffer with smooth decay
        self.trail_buffer = self.trail_buffer * (1 - self.config.blur_factor) + new_grid * self.config.blur_factor
        
        self.grid = new_grid
        self.frame_count += 1
        
        # Blend current grid with trails
        return np.maximum(self.grid, self.trail_buffer * 0.5)

class LeniaVisualizer:
    def __init__(self, lenia: Lenia, interval: int = 50):
        self.lenia = lenia
        self.interval = interval
        
        # Setup plot with improved styling
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(8, 8))
        self.grid_ax = plt.subplot(111)
        
        # Remove axes for cleaner look
        self.grid_ax.set_xticks([])
        self.grid_ax.set_yticks([])
        
        # Initialize grid display with enhanced colormap
        self.img = self.grid_ax.imshow(
            lenia.grid,
            cmap='magma',
            interpolation='nearest'  # Changed from 'gaussian' to 'nearest' for better performance
        )
        
        # Add subtle grid lines
        self.grid_ax.grid(False)
        
        # Custom title styling
        self.title = self.grid_ax.set_title(
            'Lenia Evolution - Generation 0',
            color='w',
            pad=10
        )
        
        # Add colorbar for reference
        plt.colorbar(self.img, ax=self.grid_ax, fraction=0.046, pad=0.04)
        
    def update(self, frame):
        """Update visualization with enhanced effects."""
        grid = self.lenia.update()
        self.img.set_array(grid)
        self.title.set_text(f'Lenia Evolution - Generation {self.lenia.frame_count}')
        
        return [self.img, self.title]
        
    def animate(self, frames: int = 1000):
        """Create and display the animation."""
        ani = FuncAnimation(
            self.fig,
            self.update,
            frames=frames,
            interval=self.interval,
        )
        return ani

def main():
    # Create configuration with enhanced visual parameters
    config = LeniaConfig(
        size=50,
        initial_live_probability=0.05,
        time_step=0.05,        # Slower time step
        kernel_radius=5,
        growth_center=0.135,   # Adjusted growth center
        growth_width=0.025,    # Wider growth width
        kernel_rings=2,
        save_frames=False,
        color_scheme='plasma',
        blur_factor=0.2
    )
    
    lenia = Lenia(config)
    visualizer = LeniaVisualizer(lenia)
    ani = visualizer.animate(frames=1000)
    
    plt.show()

if __name__ == "__main__":
    main()