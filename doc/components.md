# Components:

## 1. Simulator(ODE)
    Inputs:
    simulation discretization (float)

    control discretization (float)

    relative tolerance for solve_ivp (float)

    absolute tolerance for solve_ivp (float)

    initial condition of the system (np.array)

    dynamics function for solve_ivp (fcn)

    Outputs:
    System advanced one timestep into the future (np.array)

    Components used: scipy.integrate.solve_ivp

    Side effects: Keep track of current state of the system (change object attribute).

## 2. Simuate closed loop
    Inputs:
    trained forecaster (ESNForecaster, LSTMForecaster, etc.)

    instantiated controller object with control weightings (MPController)

    simulation environment(ODE)

    Outputs:
    simulated closed loop trajectory (np.array)

    Side effects: None.

## 3. (Experimental) S5 model implemented
    Inputs: 
    dimensionality of S5 layers (int)

    number of stacked S5 layers (int)

    input dimension (int, dim. control input + dim. state measurement)

    Outputs:
    initialized model (nn.Module)

    Side effects:
    update parameters of S5 model

    Components used: nn.Module

## 4. Forecast (S5 forward)

    Inputs:
    time history of measurements (torch.Tensor)

    current sensor measurements (torch.Tensor)

    delta (float)

    Outputs: 
    forecast of system dynamics (torch.Tensor)

    Components used:
    module forward function

    No side effects.

Caveat: Pytorch implementation is going to be much slower than it could be in Jax.

## 5. (Experimental) Train S5 model
    Inputs:
    model (nn.Module)

    training data (torch.tensor)

    learning rate (float)

    lags (int)

    num_epochs (int)

    Outputs: 
    None

    Side effects: 
    update parameters of S5 model

    Components used:
    S5 model