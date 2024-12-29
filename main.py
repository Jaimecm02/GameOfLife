import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update(frame_num, img, grid, size):
    # Create a copy of the grid to calculate the next state
    new_grid = grid.copy()
    for row in range(size):
        for col in range(size):
            # Count live neighbors
            live_neighbors = (
                grid[row, (col-1)%size] + grid[row, (col+1)%size] +
                grid[(row-1)%size, col] + grid[(row+1)%size, col] +
                grid[(row-1)%size, (col-1)%size] + grid[(row-1)%size, (col+1)%size] +
                grid[(row+1)%size, (col-1)%size] + grid[(row+1)%size, (col+1)%size]
            )
            # Apply Conway's rules
            if grid[row, col] == 1:  # Live cell
                if live_neighbors < 2 or live_neighbors > 3:
                    new_grid[row, col] = 0  # Dies
            else:  # Dead cell
                if live_neighbors == 3:
                    new_grid[row, col] = 1  # Becomes alive
    # Update the data of the grid
    img.set_data(new_grid)
    grid[:] = new_grid[:]
    return img,

def main():
    # Set grid size
    size = 100
    # Initialize grid with random 0s and 1s
    grid = np.random.choice([0, 1], size=(size, size))
    
    # Set up the figure and animation
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest', cmap='binary')
    ani = animation.FuncAnimation(fig, update, fargs=(img, grid, size), 
                                  frames=100, interval=100, save_count=50)
    plt.show()

if __name__ == '__main__':
    main()
