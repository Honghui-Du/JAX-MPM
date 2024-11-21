# import taichi as ti
import matplotlib.pyplot as plt
import jax.numpy as np
import jax
# import tkinter as tk
import jax.scipy.linalg as jsl
from jax import random, vmap, grad, jit
import taichi as ti
import time
import imageio
# import imageio
# from functools import partial
from jax.numpy.linalg import svd
import numpy as nnp

from scipy.io import loadmat



n_grid_x = 300

data = loadmat('test_' + str(n_grid_x) + '.mat')
Xi = data['node']
Xi_rows = data['xp']
n_particles=Xi_rows.shape[0]


dim = 2

# n_grid_x = 100
dh = 3.0 / n_grid_x

n_grid_y = int(2.0/dh)

inv_dh = 1/dh

dt = 1e-4
p_vol_orginal, p_rho_orginal = (dh * 0.5)*(dh * 0.5), 1000
p_mass = p_vol_orginal * p_rho_orginal
rho_ref = 1e3
c = 50
# c = 10
mu = 1.01*1e-3
gravity = 9.8

max_steps = 100000
real_time = 5
steps  = round(real_time/dt) +1


x = np.zeros((n_particles, dim))
# x_save = np.zeros((steps, n_particles, dim))

v = np.zeros((n_particles, dim))

C = np.zeros((n_particles, dim, dim))
F = np.zeros((n_particles, dim, dim))

pressure = np.zeros((n_particles, dim, dim))

init_v = np.zeros(dim)


bound = 5
# loss = np.zeros()




@jit
def pre_compute(x):

    base = np.trunc(x * inv_dh - 0.5).astype(np.int32)
    fx = x * inv_dh - base.astype(np.int32)

    w = np.array([0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2])
    return base, fx, w

    
@jit
def p2g(f: np.int32, F, v, x, C, grid_v_in, grid_m_in, base, fx, w):


    new_F = (np.eye(2) + dt * C) @ F

    F = F.at[:].set(new_F)

    J = np.linalg.det(new_F)

    p_vol_next = p_vol_orginal * J
    p_rho_next = p_rho_orginal / J

    pressure_each = c**2*(p_rho_next - p_rho_orginal)
    # pressure_each = c**2*p_rho_orginal*(J - 1)
    


    pressure = pressure_each[:, None, None] * np.eye(2) 

    stress = - pressure  \
        - 2/3 * mu * np.trace(C, axis1=1, axis2=2)[:, None, None] * np.eye(2) \
            + mu * (C + C.transpose(0, 2, 1))
    
    stress = (-dt * 4  * p_vol_next[:, None, None] * inv_dh * inv_dh) * stress
    
    affine = stress + p_mass * C

    # Grid update - needs to be further optimized or adapted for JAX
    # cece=[]
    for i in range(3):
        for j in range(3):
            offset = np.array([i, j])
            dpos = (np.array([i, j], dtype=float) - fx) * dh
            weight = w[i, :, 0] * w[j, :, 1]
            grid_v_in = grid_v_in.at[(base + offset)[:,0], (base + offset)[:,1],:].add(weight[:, None] * (p_mass * v + np.einsum('ijk,ik->ij', affine, dpos)))
            grid_m_in = grid_m_in.at[(base + offset)[:,0], (base + offset)[:,1]].add(weight * p_mass)

    return F, grid_v_in, grid_m_in


@jit
def grid_op(f: np.int32, grid_v_in, grid_m_in):

    # Compute the inverse of the mass matrix with a small epsilon to avoid division by zero
    inv_m = 1 / (grid_m_in + 1e-10)
    
    # Update velocities
    v_out = inv_m[:, :, None] * grid_v_in  # Broadcasting to apply inv_m to each component
    v_out = v_out.at[:, :, 1].add(-dt * gravity)  # Apply gravity to the second component of velocity

    # Boundary conditions

    mask_left = (np.arange(n_grid_x) < bound)[:, None] * (v_out[:, :, 0] < 0)
    mask_right = (np.arange(n_grid_x) > (n_grid_x - bound))[:, None] * (v_out[:, :, 0] > 0)
    mask_bottom = (np.arange(n_grid_y) < bound) * (v_out[:, :, 1] < 0)
    mask_top = (np.arange(n_grid_y) > (n_grid_y - bound)) * (v_out[:, :, 1] > 0)


    v_out_x = np.where((mask_left | mask_right), 0, v_out[:,:,0])
    v_out_y = np.where((mask_bottom | mask_top), 0, v_out[:,:,1])

    grid_v_out = np.stack((v_out_x, v_out_y), axis=-1)

    # Output updated velocities
    return grid_v_out


@jit
def g2p(f: np.int32, grid_v_out, v, x, base, fx, w):


    new_v = np.zeros((n_particles, dim))
    new_C = np.zeros((n_particles, dim, dim))

    for i in range(3):
        for j in range(3):
            # offset = np.array([i, j])
            dpos = (np.array([i, j], dtype=float) - fx)
            g_v = grid_v_out[base[:,0] + i, base[:,1] + j]
            weight = w[i, :, 0] * w[j, :, 1]

            new_v += weight[:, None] * g_v

            new_C += 4 * weight[:, None, None]  * (np.einsum('ij,ik->ijk', g_v, dpos) * inv_dh)

    v = new_v
    x = x + dt * v
    C = new_C
    return v, x, C





@jit
def substep(s, F, v, x, C):

    grid_v_in = np.zeros((n_grid_x, n_grid_y, dim))
    grid_m_in = np.zeros((n_grid_x, n_grid_y))
    
    base, fx, w = pre_compute(x)

    # start_time = time.time()
    F, grid_v_in, grid_m_in = p2g(s, F, v, x, C, grid_v_in, grid_m_in, base, fx, w)
    # p2g_time = time.time() - start_time

    # start_time = time.time()
    grid_v_out = grid_op(s, grid_v_in, grid_m_in)
    # grid_op_time = time.time() - start_time

    # start_time = time.time()
    v, x, C = g2p(s, grid_v_out, v, x, base, fx, w)
    # g2p_time = time.time() - start_time

    return F, v, x, C



identity_matrix = np.identity(2)
F = np.tile(identity_matrix, (n_particles, 1, 1))



x = np.array(Xi_rows)+ (bound - 1) * dh

# x_save = x_save.at[0].set(x)

v = np.tile(init_v, (n_particles, 1))

losses = []
img_count = 0




x_min, x_max = 0.0, 3.0
y_min, y_max = 0.0, 1.6
def normalize(points, x_min, x_max, y_min, y_max):
    # Normalize x and y independently
    points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)  # Normalize x
    points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)  # Normalize y
    return points

gui = ti.GUI("Explicit MPM", res=(640, 640), background_color=0x112F41)

start_time = time.time()
frames = []
ii = -1
t=0
for step in range(steps):
    ft = 0.01*(step+1)
    while t < ft + 1e-10:
        ii=ii+1
        F, v, x, C = substep(ii, F, v, x, C)
        # F, v, x, C, x_save = test_steps(steps, F, v, x, C, grid_v_in, grid_m_in, x_save)
        t = t+dt
    if ii%5000==0:
        elapse_time = time.time()-start_time
        print(t,elapse_time)
    # colors = np.array([0xED553B,0x068587,0xEEEEF0], dtype=np.uint32)
    # x_plot=x.to_numpy()
    # x_plot = normalize(x_plot, x_min, x_max, y_min, y_max)
    # gui.circles(x_plot, radius=1.5, color=colors[material.to_numpy()])


    colors = nnp.array([0xED553B,0x068587,0xEEEEF0], dtype=np.uint32)
    x_plot=nnp.array(x)
    x_plot = normalize(x_plot, x_min, x_max, y_min, y_max)
    gui.circles(x_plot, radius=1.5, color=colors[np.zeros(n_particles, dtype=np.int32)])

    frame = gui.get_image()
    frame = (frame * 255).astype(np.uint8)
    frames.append(frame)
    gui.show() # Change to gui.show(f'{frame:06d}.png') to write images to disk

imageio.mimsave('simulation.gif', frames, fps=10)

