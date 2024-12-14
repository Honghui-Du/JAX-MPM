import matplotlib.pyplot as plt
import jax.numpy as np
import jax

from jax import jit

import time

import numpy as nnp


Mu = 0.2
A = 0.01
dim = 3
dt = 0.0001
save_interval = 200

massLengthA = 0.1
massLengthB = np.sqrt(2 * massLengthA * massLengthA)
massK      = 20000.0

pointMass = 0.2

widthSize,heightSize = 127, 127

faceSize = widthSize * heightSize * 2

pointSize = (widthSize + 1) * (heightSize + 1)


pointLocation = np.zeros((pointSize, dim))
pointVelocity = np.zeros((pointSize, dim))



Idx = np.zeros(faceSize * 3, dtype=np.int32)

G = np.array([0.0, 0.0, -9.8], dtype=np.float32)
Wind = np.array([3.0, 2.0, 1.0], dtype=np.float32)


def wind_model(t, omega, phi, k, d):
    wind_force = (np.sin(omega * t + phi) + 1.0) / 2
    return wind_force * k[:, np.newaxis] * d[np.newaxis, :]

def wind_model_multiple(t, omega, phi, k, d):
    wind_force = (np.sin(omega * t + phi) + 1.0) / 2
    return wind_force[:, np.newaxis, np.newaxis] * k[:, np.newaxis] * d[np.newaxis, :]




indices = []
for r in range(widthSize):
    for c in range(heightSize):
        top_left = r * (widthSize + 1) + c
        top_right = top_left + 1
        bottom_left = (r + 1) * (heightSize + 1) + c
        bottom_right = bottom_left + 1
        indices.append([top_left, bottom_left, top_right])  # First triangle
        indices.append([top_right, bottom_left, bottom_right])  # Second triangle
indices = nnp.array(indices, dtype=np.int32)
faces = nnp.hstack([[3] + list(triangle) for triangle in indices])



@jit
def pointID(x, y):
    # Check bounds using logical conditions
    in_bounds = (x >= 0) & (x <= widthSize) & (y >= 0) & (y <= heightSize)
    # Compute ID only for valid coordinates
    R = np.where(in_bounds, y * (widthSize + 1) + x, -1)
    return R


def pointCoord(ID):
    # Compute x and y coordinates
    x = ID % (widthSize + 1)
    y = ID // (widthSize + 1)
    return (x, y)

def init_jax():
    # Create a batch of indices
    indices = np.arange(pointSize)


    x, y = pointCoord(indices)

    # Initialize pointLocation with calculated x, y, and fixed z = 6
    pointLocation = np.stack((x * massLengthA, y * massLengthA, np.full_like(x, 6.0)), axis=-1)
    # pointLocation = np.stack((np.full_like(x, 0.0), x * massLengthA, y * massLengthA), axis=-1)
    R_y = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0,  0.0],
    [-1.0, 0.0,  0.0]
    ])
    pointLocation = np.dot(R_y, pointLocation.T).T
    # Initialize pointVelocity with zeros
    pointVelocity = np.zeros((pointSize, 3), dtype=np.float32)
    

    return pointLocation, pointVelocity

@jit
def massID(ID):

    x, y = pointCoord(ID)

    # Update the neighbor IDs
    deltas = np.array([
        [-1, 0],  # left
        [1, 0],   # right
        [0, -1],  # down
        [0, 1],   # up
        [-1, -1], # bottom-left
        [1, 1],   # top-right
        [-1, 1],  # top-left
        [1, -1]   # bottom-right
    ])

    # Broadcast x and y for neighbor calculation, resulting in shape (batch_size, 8)
    x_neighbors = x[:, None] + deltas[:, 0]
    y_neighbors = y[:, None] + deltas[:, 1]

    # Calculate neighbor IDs for the batch
    neighbor_IDs = pointID(x_neighbors, y_neighbors)  # Assume pointID can take vectorized x, y values

    return neighbor_IDs



@jit
def ComputeForce(pointLocation, pointVelocity, Wind):

    pointForce = np.zeros((pointSize, dim))

    num_points = pointForce.shape[0]
    

    for j in range(0, 4):
        # Mask for valid neighbors (Dirs != -1)
        # valid_mask = Dirs[:, j] != -1
        current = valid_neig[j]
        # Calculate direction vectors only for valid neighbors
        Dir = pointLocation[Dirs[:, j]] - pointLocation[np.arange(num_points)]
        Dir_norm = np.linalg.norm(Dir[current], axis=-1)

        # Calculate forces for valid neighbors
        force_update = (Dir_norm - massLengthA)[:, None] * massK * Dir[current] / (Dir_norm[:, None])
        # print(j, force_update[-1])
        pointForce = pointForce.at[current].add(force_update)

    # Second set of neighbors (4 to 7)
    for j in range(4, 8):
        # Mask for valid neighbors (Dirs != -1)
        # valid_mask = Dirs[:, j] != -1
        current = valid_neig[j]
        # Calculate direction vectors only for valid neighbors
        Dir = pointLocation[Dirs[:, j]] - pointLocation[np.arange(num_points)]
        Dir_norm = np.linalg.norm(Dir[current], axis=-1)

        # Calculate forces for valid neighbors
        force_update = (Dir_norm - massLengthB)[:, None] * massK * Dir[current] / (Dir_norm[:, None])
        # print(j, force_update[-1])
        # Apply the update only for valid neighbors
        pointForce = pointForce.at[current].add(force_update)

    # Add gravitational and wind forces
    pointForce += G * pointMass + Wind

    # Add velocity-related forces
    pointForce += A * pointVelocity * pointVelocity

    # Apply boundary conditions
    pointForce = pointForce.at[holding_index[0]].set(np.array([0.0, 0.0, 0.0]))
    pointForce = pointForce.at[holding_index[1]].set(np.array([0.0, 0.0, 0.0]))

    return pointForce

@jit
def forward_jax(pointForce, pointLocation, pointVelocity, T):
    # Update pointVelocity: pointVelocity[i] += T * pointForce[i] / pointMass
    pointVelocity = pointVelocity + (T * pointForce / pointMass)

    # Update pointLocation: pointLocation[i] += T * pointVelocity[i]
    pointLocation = pointLocation + (T * pointVelocity)

    return pointLocation, pointVelocity

def Init():
    # Create grid indices for width and height
    pointLocation, pointVelocity = init_jax()

    i_vals = np.arange(widthSize)
    j_vals = np.arange(heightSize)
    i_grid, j_grid = np.meshgrid(i_vals, j_vals, indexing='ij')

    # Flatten the grids to create all combinations of (i, j)
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    # Calculate point IDs for each corner of the small rectangles
    ID_1 = pointID(i_flat, j_flat)
    ID_2 = pointID(i_flat + 1, j_flat)
    ID_3 = pointID(i_flat, j_flat + 1)
    ID_4 = pointID(i_flat + 1, j_flat + 1)

    # Construct the Idx array
    Idx = np.zeros((widthSize * heightSize * 6,), dtype=np.int32)

    # Fill in the Idx array using JAX operations
    Idx = Idx.at[0::6].set(ID_1)
    Idx = Idx.at[1::6].set(ID_2)
    Idx = Idx.at[2::6].set(ID_3)
    Idx = Idx.at[3::6].set(ID_2)
    Idx = Idx.at[4::6].set(ID_3)
    Idx = Idx.at[5::6].set(ID_4)

    return pointLocation, pointVelocity, Idx

def record_location(recordedLocations, t, pointLocation):
    return recordedLocations.at[t // save_interval].set(pointLocation)

@jit
def Step(pointLocation, pointVelocity):
    
    for ii in range(50):
        pointForce = ComputeForce(pointLocation, pointVelocity, Wind)
        pointLocation, pointVelocity = forward_jax(pointForce, pointLocation, pointVelocity, dt)


    return pointLocation, pointVelocity



pointLocation, pointVelocity, Idx = Init()


Dirs = massID(np.arange(pointSize))
valid_mask = np.zeros((pointSize,8))
valid_neig = []
for j in range(0, 8):
    valid_mask = valid_mask.at[:,j].set(Dirs[:, j] != -1)
    valid_neig.append(np.where(valid_mask[:, j])[0])



holding_index = [pointID(0, 0), pointID(0, heightSize)]



Frame = 0



import pyvista as pv
# import numpy as np

# Create point data for visualization

plotter = pv.Plotter()

mesh = pv.PolyData(nnp.array(pointLocation), faces=faces)
plotter.add_mesh(mesh, color="lightblue", show_edges=True, smooth_shading=True)

# Add a light source for better visualization
plotter.add_light(pv.Light(position=(5, 5, 5), intensity=1.0, color=[1, 1, 1]))

holding_points = [pointLocation[pointID(0, 0)], pointLocation[pointID(0, heightSize)]]
for point in holding_points:
    sphere = pv.Sphere(radius=0.2, center=point)  # Create a sphere at the point
    plotter.add_mesh(sphere, color="red")  # Add the sphere to the plotter


plotter.reset_camera()

plotter.camera.zoom(0.75)

plotter.camera.position =  (24.80282109587461, -23.10921715805012, 1.9165020955640446)
plotter.camera.focal_point = (3.0821756068617105, 6.34999967366457 + 3, -6.249999903142452)
# plotter.show(auto_close=False)
plotter.show(interactive_update=True)
plotter.open_gif("cloth_simulation.gif")  # Optional: create an animation

frame = 0

num_steps = 10000*3 + 1


for iter in range(num_steps):

    ##########################################
    pointLocation, pointVelocity = Step(pointLocation, pointVelocity)


    mesh.points = nnp.array(pointLocation)
    plotter.update()  # Refresh the visualization
    plotter.render()

    plotter.write_frame()
    ##########################################
plotter.close()



