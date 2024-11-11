import numpy as np
import scipy
import mcubes
import os

class Simulator:
    def __init__(self, grid_size, cell_extent, reconstruct_resolution, reconstruct_radius, reconstruct_threshold, result_dir='output', mode='mesh', cur_step=0, t=0.0):
        # Initialize grid parameters
        self.grid_size = np.array(grid_size, dtype=np.int32)  # Size of the grid (number of cells along each dimension)
        self.cell_extent = cell_extent  # Physical size of a grid cell
        self.grid_extent = self.grid_size * self.cell_extent  # Total physical size of the entire grid
        self.dx = self.cell_extent  # Grid resolution

        # Initialize particle information
        # self.particles_position = particles_position  # Positions of particles (N, 3) NumPy array

        # Reconstruction settings
        self.reconstruct_resolution = reconstruct_resolution  # Resolution for reconstructing the scalar field
        self.reconstruct_radius = reconstruct_radius  # Radius of influence for each particle in the scalar field
        self.reconstruct_threshold = reconstruct_threshold  # Threshold for marching cubes

        # Other settings
        self.result_dir = result_dir  # Directory to save the mesh
        self.mode = mode  # Mode (useful for naming files)
        self.cur_step = cur_step  # Current step of the simulation
        self.t = t  # Time of the simulation (useful for naming files)

    def metaball_scalar_field(self, resolution, particle_pos):
        # Compute the grid step sizes
        dx = self.grid_extent[0] / resolution[0]
        dy = self.grid_extent[1] / resolution[1]
        dz = self.grid_extent[2] / resolution[2]

        # Set the influence radius for particles
        radius = self.reconstruct_radius

        # Convert particle positions to a NumPy array
        # particle_pos = self.particles_position

        # Create a KDTree to find nearby particles efficiently
        kdtree = scipy.spatial.KDTree(particle_pos)

        # Create an expanded 3D grid
        f_shape = (resolution[0] + 3, resolution[1] + 3, resolution[2] + 3)
        i, j, k = np.mgrid[:f_shape[0], :f_shape[1], :f_shape[2]] - 1

        # Calculate physical positions of grid points
        x = i * dx
        y = j * dy
        z = k * dz
        field_pos = np.stack((x, y, z), axis=3)

        # Find nearby particles for each grid point using KDTree
        particle_indices = kdtree.query_ball_point(field_pos, radius, workers=-1)

        # Initialize the scalar field
        f = np.zeros(f_shape)

        # Function to evaluate metaball influence
        def metaball_eval(p1, p2s):
            if p2s is None:
                return 0.0
            r = np.clip(np.linalg.norm(p1 - p2s, axis=1) / radius, 0.0, 1.0)
            return (1.0 - r ** 3 * (r * (r * 6.0 - 15.0) + 10.0)).sum()

        # Iterate over grid points and compute the scalar field
        for ii in range(resolution[0]):
            for jj in range(resolution[1]):
                for kk in range(resolution[2]):
                    idx = (ii, jj, kk)
                    p1 = np.array([x[idx], y[idx], z[idx]])
                    p2s = np.array([particle_pos[p2_idx] for p2_idx in particle_indices[idx]]) if particle_indices[idx] else None
                    f[idx] = metaball_eval(p1, p2s)
        
        return f

    def reconstruct_mesh(self, filename, particle_position):
        # Generate the scalar field from particles
        f = self.metaball_scalar_field(self.reconstruct_resolution, particle_position)

        # Use Marching Cubes to reconstruct the mesh
        vertices, triangles = mcubes.marching_cubes(f, self.reconstruct_threshold)

        # Ensure the result directory exists
        os.makedirs(self.result_dir, exist_ok=True)

        # Define the filename if none is provided
        if filename is None:
            filename = '{0}_{1:04d}_{2}s.obj'.format(self.mode, self.cur_step, round(self.t, 3))

        path = os.path.join(self.result_dir, filename)

        # Export the mesh to an OBJ file
        mcubes.export_obj(vertices, triangles, path)

        print(f"Mesh has been saved as an OBJ file: {path}")