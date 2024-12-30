import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import convolve2d
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class GridConfig:
    """Configuration settings for a single grid."""
    size: int
    initial_live_probability: float
    colormap: LinearSegmentedColormap
    alpha: float = 0.5

class GameState:
    """Manages the state and evolution of a single Game of Life grid."""
    def __init__(self, config: GridConfig):
        self.config = config
        self.grid = self._initialize_grid()
        self.previous_states: List[np.ndarray] = []
        self.max_history = 3  # Number of previous states to keep for motion blur
        
    def _initialize_grid(self) -> np.ndarray:
        return np.random.choice(
            [0, 1],
            size=(self.config.size, self.config.size),
            p=[1 - self.config.initial_live_probability, self.config.initial_live_probability]
        )
    
    def update(self) -> np.ndarray:
        """Update the grid according to Game of Life rules."""
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
        
        live_neighbors = convolve2d(self.grid, kernel, mode='same', boundary='wrap')
        new_grid = (self.grid & ((live_neighbors == 2) | (live_neighbors == 3))) | \
                  (~self.grid & (live_neighbors == 3))
        
        # Update motion blur history
        self.previous_states.append(self.grid.copy())
        if len(self.previous_states) > self.max_history:
            self.previous_states.pop(0)
            
        self.grid = new_grid
        return self._calculate_motion_blur()
    
    def _calculate_motion_blur(self) -> np.ndarray:
        """Calculate motion blur effect from previous states."""
        blurred_grid = np.zeros_like(self.grid, dtype=float)
        for idx, previous_grid in enumerate(self.previous_states):
            weight = (idx + 1) / len(self.previous_states)
            blurred_grid += previous_grid * weight
        return blurred_grid

class ColorScheme:
    """Manages color schemes and gradients for visualization."""
    @staticmethod
    def create_neon_colormaps() -> List[LinearSegmentedColormap]:
        neon_colors = [
            ['#ffffff', '#FF00FF'],  # Neon purple
            ['#ffffff', '#00FF00'],  # Neon green
            ['#ffffff', '#00FFFF'],  # Neon cyan
            ['#ffffff', '#FF0000']   # Neon red
        ]
        return [LinearSegmentedColormap.from_list('custom', colors) for colors in neon_colors]
    
    @staticmethod
    def create_gradient_background(size: int, direction: str = 'horizontal') -> np.ndarray:
        """Create a gradient background for the grid."""
        if direction == 'horizontal':
            gradient = np.linspace(0, 1, size).reshape(1, -1)
            return np.tile(gradient, (size, 1))
        elif direction == 'vertical':
            gradient = np.linspace(0, 1, size).reshape(-1, 1)
            return np.tile(gradient, (1, size))
        elif direction == 'radial':
            x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
            gradient = np.sqrt(x**2 + y**2)
            return gradient / gradient.max()
        else:
            raise ValueError("Invalid direction. Choose 'horizontal', 'vertical', or 'radial'.")

class GameOfLifeSimulation:
    """Manages multiple Game of Life grids and their visualization."""
    def __init__(self, 
                 size: int = 100,
                 num_grids: int = 4,
                 live_probability: float = 0.2):
        self.size = size
        self.num_grids = num_grids
        
        # Initialize color schemes
        self.colormaps = ColorScheme.create_neon_colormaps()
        
        # Create grid configurations
        self.grid_configs = [
            GridConfig(size=size,
                      initial_live_probability=live_probability,
                      colormap=self.colormaps[i])
            for i in range(num_grids)
        ]
        
        # Initialize game states
        self.games = [GameState(config) for config in self.grid_configs]
        
        # Setup visualization
        self.fig, self.ax = plt.subplots()
        self.setup_visualization()
        
    def setup_visualization(self):
        """Set up the visualization components."""
        # Create and display gradient background
        gradient = ColorScheme.create_gradient_background(self.size, 'radial')
        self.ax.imshow(gradient, interpolation='nearest', cmap='cool', alpha=0.8, zorder=0)
        
        # Create image objects for each grid
        self.img_list = []
        for i, game in enumerate(self.games):
            img = self.ax.imshow(game.grid,
                               interpolation='nearest',
                               cmap=game.config.colormap,
                               alpha=game.config.alpha,
                               zorder=1 + i)
            self.img_list.append(img)
        
        # Remove axes for cleaner look
        self.ax.set_xticks([])
        self.ax.set_yticks([])
    
    def update(self, frame_num: int) -> List:
        """Update function for animation."""
        plt.title(f'Generation {frame_num}')
        
        for game, img in zip(self.games, self.img_list):
            blurred_grid = game.update()
            img.set_data(blurred_grid)
            
        return self.img_list
    
    def animate(self, interval: int = 100):
        """Start the animation."""
        self.animation = animation.FuncAnimation(
            self.fig,
            self.update,
            interval=interval,
            cache_frame_data=False
        )
        plt.show()

def main():
    # Create and run simulation
    simulation = GameOfLifeSimulation(
        size=100,
        num_grids=4,
        live_probability=0.2
    )
    simulation.animate()

if __name__ == '__main__':
    main()