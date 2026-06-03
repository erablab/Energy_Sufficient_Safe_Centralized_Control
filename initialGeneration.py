import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
# Configuration
num_particles = 10
sigma = 0.2
# Compute the mask based on distance from the center
particles = []
L = 10.0

# Set up the figure
fig, ax = plt.subplots()
ax.set_xlim(-L/2, L/2)
ax.set_ylim(-L/2, L/2)
ax.set_xticks(np.arange(-L/2, L/2, 0.5))
ax.set_yticks(np.arange(-L/2, L/2, 0.5))
ax.grid(True)
ax.set_title(f"Click to place {num_particles} particles")




# Callback function for mouse clicks
def on_click(event):
    global particles
    if event.inaxes is not None and len(particles) < num_particles:
        x, y = event.xdata, event.ydata

        particles.append({"position": [x, y], "sigma": sigma})
        ax.plot(x, y, 'bo')  # Mark point in blue
        plt.draw()

        if len(particles) == num_particles:
            save_particles()


# Function to save particles to a JSON file
def save_particles():
    with open("initial_particles.json", "w") as file:
        json.dump(particles, file)
    print("Particle positions saved to 'initial_particles.json'.")
    plt.close()


# Connect the event listener
fig.canvas.mpl_connect("button_press_event", on_click)

# Show the interactive plot
plt.show()
