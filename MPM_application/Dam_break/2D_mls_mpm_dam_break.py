import matplotlib.pyplot as plt
import jax.numpy as np
from jax import jit
import time
import numpy as nnp






cal_x, cal_y = 3.0, 1.5

n_grid_x = 200

dim = 2

dh = cal_x / n_grid_x

n_grid_y = int(cal_y/dh)



real_x, real_y = 1.2, 0.6


density = 2
w_count = int(n_grid_x/(cal_x/real_x) * density)
h_count = int(n_grid_y/(cal_y/real_y) * density)


real_dx = real_x / w_count
real_dy = real_y / h_count

x_init, y_init = 0, 0
x = []
n_particles = 0
for i in range(w_count):
    for j in range(h_count):
            x.append([
                x_init  + (i) * real_dx + 0,
                y_init  + (j) * real_dy + 0,
            ])
            n_particles += 1

x = np.array(x)


inv_dh = 1/dh

dt = 1e-5
p_vol_orginal, p_rho_orginal = (dh * 0.5)*(dh * 0.5), 1000
p_mass = p_vol_orginal * p_rho_orginal
rho_ref = 1e3
c = 50
# c = 10
mu = 1.01*1e-3
gravity = 9.8


steps  = 1000


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
def p2g(F, v, x, C, grid_v_in, grid_m_in, base, fx, w):


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
def grid_op(grid_v_in, grid_m_in):

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
def g2p(grid_v_out, v, x, base, fx, w):


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
def substep(F, v, x, C):

    grid_v_in = np.zeros((n_grid_x, n_grid_y, dim))
    grid_m_in = np.zeros((n_grid_x, n_grid_y))
    
    base, fx, w = pre_compute(x)

    # start_time = time.time()
    F, grid_v_in, grid_m_in = p2g(F, v, x, C, grid_v_in, grid_m_in, base, fx, w)
    # p2g_time = time.time() - start_time

    # start_time = time.time()
    grid_v_out = grid_op(grid_v_in, grid_m_in)
    # grid_op_time = time.time() - start_time

    # start_time = time.time()
    v, x, C = g2p(grid_v_out, v, x, base, fx, w)
    # g2p_time = time.time() - start_time

    return F, v, x, C



identity_matrix = np.identity(2)
F = np.tile(identity_matrix, (n_particles, 1, 1))



x = x + (bound - 1) * dh


v = np.tile(init_v, (n_particles, 1))

losses = []
img_count = 0




x_min, x_max = 0.0, cal_x
y_min, y_max = 0.0, cal_y


import pyvista as pv
plotter = pv.Plotter()



corners = nnp.array([[x_min, y_min, 0],
                    [x_max, y_min, 0],
                    [x_max, y_max, 0],
                    [x_min, y_max, 0]])

# Create a PolyData object
frame = pv.PolyData(corners)

# Add lines between the corners to form a rectangle
lines = nnp.array([2, 0, 1, 2, 1, 2, 2, 2, 3, 2, 3, 0])
frame.lines = lines


plotter.add_mesh(frame, color='black', line_width=1)

plotter.add_light(pv.Light(position=(5, 5, 5), intensity=1.0, color=[1, 1, 1]))
points = nnp.hstack((x, np.zeros((n_particles, 1))))
point_cloud = pv.PolyData(points)
plotter.add_mesh(point_cloud, color="blue", point_size=3, render_points_as_spheres=True)
plotter.image_transparent_background = True  # Enable transparent background
plotter.view_xy()
plotter.show(interactive_update=True)

plotter.open_gif("2d_mpm_dam_break.gif")  # Optional: create an animation

start_time = time.time()
frames = []
ii = -1
t=0
for step in range(steps):
    ft = 0.01*(step+1)
    while t < ft + 1e-10:
        ii=ii+1
        F, v, x, C = substep(F, v, x, C)
        t = t+dt

    if ii%5000==0:
        elapse_time = time.time()-start_time
        print(t,elapse_time)

    points = nnp.hstack((x, np.zeros((n_particles, 1))))
    point_cloud.points = points
    plotter.update()  # Refresh the visualization
    plotter.render()
    plotter.write_frame()


plotter.close()

