# Multi-Grid Game of Life Visualization

A Python implementation of Conway's Game of Life featuring multiple simultaneous grids with neon visualization effects and motion blur.

## Features

- Multiple simultaneous Game of Life grids
- Neon color schemes with customizable gradients
- Motion blur effects for smooth visualization
- Modular and object-oriented design
- Configurable grid sizes and initial conditions

## Requirements

```
numpy
matplotlib
scipy
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multi-grid-game-of-life.git
cd multi-grid-game-of-life
```

2. Install the required packages:
```bash
pip install numpy matplotlib scipy
```

## Usage

Run the simulation with default settings:

```bash
python game_of_life.py
```

### Customization

You can customize the simulation by modifying the parameters in `main()`:

```python
simulation = GameOfLifeSimulation(
    size=100,              # Grid size (100x100)
    num_grids=4,          # Number of simultaneous grids
    live_probability=0.2   # Initial probability of live cells
)
```

## Project Structure

- `GameOfLifeSimulation`: Main class that orchestrates the simulation
- `GameState`: Manages individual grid states and evolution
- `GridConfig`: Configuration class for grid parameters
- `ColorScheme`: Handles color schemes and gradients

### Key Components

#### GridConfig
Stores configuration for each grid:
- Grid size
- Initial live cell probability
- Color scheme
- Transparency level

#### GameState
Manages the evolution of each grid:
- Grid initialization
- State updates based on Conway's rules
- Motion blur calculation
- History tracking

#### ColorScheme
Handles visualization aspects:
- Neon color schemes
- Background gradients
- Custom colormaps

## Game of Life Rules

The simulation follows Conway's Game of Life rules:

1. Any live cell with fewer than two live neighbors dies (underpopulation)
2. Any live cell with two or three live neighbors lives
3. Any live cell with more than three live neighbors dies (overpopulation)
4. Any dead cell with exactly three live neighbors becomes alive (reproduction)

## Contributing

Feel free to submit issues and enhancement requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.