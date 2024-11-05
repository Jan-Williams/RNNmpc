## Components:

1. Forecast. 
Inputs:
current state of the system (torch.DoubleTensor)
current sensor measurements (torch.DoubleTensor)
subsequent control inputs (torch.DoubleTensor)

Outputs: 
forecast of system dynamics (torch.DoubleTensor)

Components used:
module forward function (essentially just wrapped in a for loop)

No side effects.

2. Simulator(ODE)
Inputs:
simulation discretization (float)
control discretization (float)
relative tolerance for solve_ivp (float)
absolute tolerance for solve_ivp (float)
initial condition of the system (np.array)
dynamics function for solve_ivp (fcn)

Outputs:
System advanced one timestep into the future (np.array)
Simulated input for control U (np.array)

Side effects: Keep track of current state of the system (change object attribute). 
