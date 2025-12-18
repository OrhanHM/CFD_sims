# CFD_sims

## Overview
This repository contains code to execute airflow simulations around solid airfoils in 2D. The basic approach used is the finite difference discretization of the Navier-Stokes equations. If you are looking to generate final state plots from simulations we have already run, see **Reproducing Results**. For information on how to run simulations of your own, consult **Simulation Guide**.


File overview:

- *Technical_Report.pdf* is a manuscript-style report containing more on our motivations, methodology behind the simulations, and core results. 
- *system_class.py*, *wing_class.py*, and *helper_functions.py* contain all the necessary functions to run aerodynamics simulations
- *simulation.py* is an example of a verbose simulation (good for individual testing)
- *naca_testing.py* runs simulations on many NACA airfoils, saving the force calculation results. The force calculation results can be compiled from their various *.txt* files into a dataframe using *process_naca_results.py*
- *state_visualization.py* is used to create visualizations of the final state and stats from a saved simulation
- *drag_validation.py* specifically reproduces the drag comparison figure in our manuscript


## Reproducing Results
Since simulations do take considerable time to run, the final states of the simulations cited in our paper are saved and uploaded. Use *state_visualization.py*
by specifying a key from the **paths** dictionary to generate a plot of the airflow pattern simulation near the wing. 

Note: Lowering the stride variables (sx, sy) can help visualize more local fluctuations in the airflow at the expense of a busier plot. 


One can locate the exact initial conditions and final force calculation details for each result in the corresponding *.txt* file in either *./Other_testing/* or *./Naca_testing/Stats/*


## Simulation Guide 
The crux of the physics is implemented in the **System** and **Wing** classes (contained in *system_class.py* and *wing_class.py*). Consult *simulation.py* for an example simulation demonstrating procedure explained in this section. Note that with our defaults of 0.1 m/s max airspeed, 800x400 grid, and 0.005 m/s/s acceleration, simulations may take 3-10+ minutes depending on your system. We recommend running on your local machine.  

### The System class
The **System** instance is initialized with:

- Physical length of simulation in meters (X_len, Y_len)
- Number of grid points in each direction (nx, ny)
- Fluid density and dynamic viscosity (rho, nu)
- Desired airspeed in m/s to ramp to before breaking (umax)

Then: 

1. Set the initial velocity and pressure fields (u, v, p) by calling System.set_ics(u0, v0, p0). Recommended to pass in np.zeros arrays for all fields
2. Set the desired body acceleration field (ax, ay) by calling System.set_body_accelerations(ax, ay). Recommended to pass zero ay and a small constant positive ax (ie. 0.005) at all fluid points.
3. Add a **Wing** object by calling System.add_wing(**kwargs) to add a Wing with the given keyword arguments to the simulation (see below for more details on howto do this)
5. After adding a wing, call System.build_jacobi_step_matrix() (required for pressure calculations) and System.set_influence_sections() (required for force calculations)

You are ready to simulate! Use a while loop checking whether the system is still ramping to the desired airspeed umax (stored in boolean System.ramping). Within the loop, alternate between calling System.motion_step() and System.pressure_step(). Add in any desired metric printouts. At any point, forces can be calculated with System.compute_forces(). 

Once the simulation is done, call System.save_state(state_path), System.save_final_stats(stats_path), and System.plot_state(**kwargs) to see/save the final results so you don't lose your work!


### The Wing class

Wings are created by interpolation through a set of control points. The **Wing** class supports two main modes of designating these control points: custom input parameters and four-digit NACA codes. 

If using custom input parameters, initialize the **Wing** instance with:

- 1D arrays of lower and upper heights (upper_height_params, lower_height_params) representing distances away from the main chord.
- 'pchip' or 'cubic' interpolation_method
- scale = True or False

It is HIGHLY recommended to visualize design first using System.plot_wing() for custom wings. 


If using a NACA codes:

- A string four-digit NACA code (naca_code)
- interpolation_method = 'pchip'. The high number of control points means that cubic spline generates non-physical artifacts in the wing boundary. 
- scale = False. Required to preserve likeness to the defined protocol of generating NACA airfoils. 


Thicknesses between 0.1 and 0.2 work best (use values on the upper end of this range only if the wing profile is smooth, like a NACA). The other parameters (chord_length, x_start, n_points, boundary_conditions, naca_points) provide added flexibility but default to good values. 

