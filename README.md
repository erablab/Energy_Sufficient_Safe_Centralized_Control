# Energy_Sufficient_Safe_Centralized_Control
Supplementary simulation code for the ITSC 2026 paper "Safe and Energy-Aware Multi-Robot Density Control via PDE-Constrained Optimization for Long-Duration Autonomy" (https://arxiv.org/abs/2604.15524)

The RRT path-planning component is adapted from: https://github.com/stephane-caron/rrt-completeness/tree/master

## Running the Simulation

Before running the simulation, tune the RRT server parameters in `rrt_persistent_pool_server_stable.py`. The default setting uses 10 parallel processes for 10 robots and can be adjusted based on the user's machine.

Start the RRT path-planning server:

```bash
python rrt_persistent_pool_server.py --host 127.0.0.1 --port 8765
```

Then, in a separate terminal, run the simulation:

```bash
python Controller_sim.py
```

The simulation first runs with animation so the user can observe the robot behavior. After the animation window is closed, three performance plots are displayed. Once these are closed, the main call block at the end of `Controller_sim.py` runs 100 simulations for statistical analysis and saves the results in `step_logs_001`.

## Visualizing and Debugging Results

After the data is generated, two helper scripts are provided.

Run:

```bash
python Step_log_visualze.py
```

to generate the best, average, or worst-case performance plots from the runs in `step_logs_001`. The selection mode can be changed around lines 280-285 of the script. The default setting plots the worst-case run.

Run:

```bash
python findmin.py
```

to identify runs where the energy level `E_i` of any robot falls below a chosen threshold. This is useful for checking rare outlier runs in the statistical simulations.

## Note on the RRT Planner

The RRT planner is stochastic, so the generated path varies between timesteps even when the robot position changes only slightly. In a real-time closed-loop simulation, this can create temporal inconsistency between consecutive timesteps, causing issues such as sudden changes in path direction, jumps in estimated cost-to-charger, or unnecessary local loops.

To improve real-time performance and temporal consistency, our implementation uses a persistent multi-process RRT server, accepts new paths only when they improve the previous path estimate, smooths the desired direction using the first several path segments, and bypasses RRT with a straight-line motion when the robot is close to the charger.

These adaptations significantly improve the behavior compared with using the vanilla RRT planner directly. However, because the planner remains stochastic, rare outlier runs can still occur. In particular, path smoothing can occasionally produce a desired direction that is less compatible with the space CBF constraint near danger regions, causing the simulation to stall and resulting in `E_i`-> 0.

With the default parameters, approximately 1-2 outlier runs were observed per 100 simulations. Increasing the RRT sampling budget to use finer path-planning parameters reduces these cases, at the cost of higher computation time. Developing a temporally stable (RRT) planner is outside the scope of this work.
