# MATH333 Final Project

This repository contains my final project for MATH333 Differential Equations class.

## Project Description

This project implements a numerical solution for a torsional system using the Runge-Kutta 4th order (RK4) method. The system models a forced torsional oscillator with damping.

## Files

- `torsion_rk4_grid.py`: Implementation of the RK4 method for solving the torsional system with grid search optimization
- `requirements.txt`: Python dependencies for the project

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the script:
   ```bash
   python torsion_rk4_grid.py
   ```

## Dependencies

- numpy: For numerical computations
- matplotlib: For plotting error grids and time series

