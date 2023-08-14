We describe the code in the readme file:

The simulation program consists of the following folders:
- elements
- equations
- examples
- geometries
- solver
- step_size_control

## elements
The folder elements has the following files:
- Taylor-Hood P2-P1 elements.
- P1-P1 stabilized elements.
- Mini-Element
Only the Taylor-Hood class is used in the experiments and tested.

## equations
The folder equations has:
- a general PDE class.
- the derived Stokes class.
- The FullStokes class is derived from the Stokes class.
Both the Stokes and the full-Stokes equations define left- and right-hand sides.
Depending on the Newton or Picard method, the left- and right-hand sides need to be defined.

## geometries
In the geometry folder is a general class Geometry. 
This class has methods to visualize the domain.
There is also the derived class BumpyBed, which generates the domain for the experiment ISMIP-HOM B.
BumpyBed3D is derived from BumpyBed and generates the domain for the experiment ISMP-HOM A.
Both domains generate copies to the left and the right of the glacier.
Both __init__ methods have the option that the original glacier has a finer resolution, and the copies have a coarser resolution. That option works only for resolution_x_direction=98. It is helpful to calculate the solution of the 3D simulation in less computation time.

## solver
The solver saves the partial differential equation as an attribute.
There are two solvers:
- LinearDirect just solves a linear problem. The default solver is the mumps-solver.
- NonlinearIterative is also derived from the Solver class.
The Newton and the Picard class are derived from the NonlinearIterative class.

## step size control
The StepSizeControl is again the main class. The derived classes are
- Exact
- ConstantStep
- Armijo
All these classes implement the step size control with the named method.
One can choose between 'norm' (residual norm) and 'functional' as the minimization function.
Using the residual norm as a minimization term for the exact step size is impossible.

## examples
In the examples folder is the file run_fullStokes.py. If FEniCS and a compatible Python3 version are installed, that file should be executable with
python3 run_fullStokes.py
