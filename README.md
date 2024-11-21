# Explicit MPM Simulation

This repository contains an implementation of an explicit Material Point Method (MPM) simulation using Python and JAX. Specifically, MLS-MPM ([https://yzhu.io/publication/mpmmls2018siggraph/](https://yzhu.io/publication/mpmmls2018siggraph/)) is implemented.  Weakly incompressible viscous fluid with EOS equation for pressure is considered.

This simulation focuses on the **dam break problem**, utilizing **explicit time integration**. However, it also has the potential for **implicit time integration** and **data assimilation** with the help of JAX. As we tested, the JAX-based MPM provides comparable computation speed with Taichi due to the JIT and batch operation for array, while holding great potential for **deep learning** integration. We are still exploring these capabilities to unlock further improvements.

## Prerequisites

To run this code, you need the following libraries:

- Taichi
- JAX
- Matplotlib
- Tkinter
- SciPy
- ImageIO
- NumPy

All of these dependencies are listed in the `requirements.txt` file. You can install them using:

```sh
pip install -r requirements.txt
```

### Dependencies

- **taichi**: Used for visualization and particle animation.
- **jax**: Provides support for differentiable programming and efficient array operations.
- **matplotlib**: Used for creating visual animations and plots.
- **tkinter**: Provides GUI elements for interactive inputs.
- **scipy**: Utilized for various numerical functions.
- **imageio**: Creates GIF animations from particle images.
- **numpy**: Provides support for numerical operations.

## Running the Simulation

The simulation begins by loading the particle data from a MATLAB `.mat` file. The parameters for the grid, time steps, particle volumes, and other properties are initialized before running the main loop of the simulation.

To run the simulation, simply execute the script using Python:

```sh
python mpm_simulation.py
```

The simulation will generate a series of images visualizing the particle movement and eventually create an animated GIF (`simulation.gif`).

## Code Structure

- **Imports**: The script makes use of several numerical and visualization libraries to load particle data, perform computations, and display animations.
- **Functions**:
  - **polar\_decomposition\_2x2**: Performs a polar decomposition on a 2x2 matrix.
  - **pre\_compute**: Computes the base indices, relative positions, and weights of particles on the grid.
  - **p2g (Particle-to-Grid)**: Transfers particle data to the grid.
  - **grid\_op**: Performs operations on the grid, including applying forces and boundary conditions.
  - **g2p (Grid-to-Particle)**: Transfers grid data back to the particles.
- **Simulation Loop**: A main loop (`substep`) updates the particles based on forces, stresses, and gravity.

## Visualization

The results are visualized in real time using Taichi's GUI module (not required), and each frame is saved to create an animated GIF. Additionally, Matplotlib is used to generate an animation of the particle movement, saved as `particle_animation.gif`.

### Notes

- The default grid size is `300 x 160` for this example.
- You can adjust parameters such as `dt` (time step), grid dimensions, and material properties to explore different simulations.

### Files

- **mpm\_simulation.py**: Main simulation script.
- **test\_300.mat**: Example particle data for running the simulation.

If you have any questions or would like more information, feel free to reach out!

