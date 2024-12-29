import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update(frame_num, img_list, grids, size):
    # Iterate over all the grids and update them
    for i, grid in enumerate(grids):
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
        # Update the grid for the next frame
        grids[i][:] = new_grid[:]
        img_list[i].set_data(new_grid)
    
    return img_list

def main():
    # Set grid size
    size = 200
    x = 0.2 # percentaje of 1s
    num_grids = 4  # Number of grids (games)
    
    # Initialize multiple grids with random 0s and 1s
    grids = [np.random.choice([0, 1], size=(size, size), p=[1-x, x]) for _ in range(num_grids)]
    
    # Set up the figure and animation
    # TODO make something with the frame_num
    fig, ax = plt.subplots()
    
    # Create an image for each grid with a different colormap
    img_list = []
    cmap_list = ['binary', 'cool', 'spring', 'plasma']  # Different color maps for each grid
    for i in range(num_grids):
        img = ax.imshow(grids[i], interpolation='nearest', cmap=cmap_list[i], alpha=0.3, zorder=i) # TODO change alpha value to depend on num_grids
        img_list.append(img)
    
    ani = animation.FuncAnimation(fig, update, fargs=(img_list, grids, size), # TODO understand all params
                                  frames=100, interval=100, save_count=50)
    plt.show()

if __name__ == '__main__':
    main()
