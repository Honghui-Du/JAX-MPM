# Physics-Based Simulation with JAX

This repository provides a suite of physics-based simulation tools using **JAX**, featuring both mass-spring and Material Point Method (MPM) examples. By leveraging JAX’s differentiable programming and just-in-time (JIT) compilation capabilities, we aim to achieve efficient and flexible simulations that can be easily integrated with deep learning frameworks.

## Key Features

- **Mass-Spring System (Cloth Simulation)**: Simulate cloth-like materials using a mass-spring system that captures realistic stretching and bending behaviors.
- **Material Point Method (MPM)**:
  - **2D Dam Break**: Weakly compressible viscous fluid simulation using the MPM approach, showcasing explicit time integration and the potential for implicit integration.
  - **3D Dam Break**: Extend the dam break scenario into three dimensions, demonstrating the versatility and scalability of the MPM implementation.

More examples and features (implicit time integration, inverse desgin, data assimilation) will be released in the future, expanding the repository’s capabilities to cover a wider range of physical scenarios.

## Prerequisites

To run the simulations, ensure that you have the following libraries installed:

- Taichi
- JAX
- Matplotlib
- Tkinter
- SciPy
- ImageIO
- NumPy

You can install these dependencies using:

```bash
pip install -r requirements.txt
