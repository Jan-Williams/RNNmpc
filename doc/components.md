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

## 3. (Experimental Class) S5 Layer
    Attributes: 

    dimensionality of S5 layers (int)

    input dimension (int)

    Methods:
    initialize state space matrix lambda
        Outputs: state space matrix

    initialize input matrix B
        Outputs: input matrix

    initialize output matrix C
        Outputs: output matrix

    discretize continuous parameters
        Outputs: lambda_bar, B_bar

    forward(u_input: torch.Tensor, delta: float)
        Outputs: output sequence

    Components used: nn.Module

## 4. (Experimental Class) S5Forecaster

    Attributes:
    input dimension(int)

    hidden dimension (int)

    number of S5 layers (int)

    number of forecast steps (int)

    output layer (torch.nn.Linear)

    delta (float)

    S5 layerlist (torch.nn.ModuleList)

    Methods: 
    forward(u_input: torch.Tensor, delta: float, x0)
        computes forecast of fcast_steps

    Components used:
    S5Layer

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
    S5Forecaster


## 6. Split train and val data
    Inputs:
    ts_data (torch.Tensor)

    lags (int)

    fcast_steps (int)

    Outputs:
    forecast of fcast_steps (torch.Tensor)

## 7. Simulator(Neuromancer)
    Inputs:
    simulation discretization (float)

    control discretization (float)

    neuromancer nonautonomous dynamics class (class)

    Outputs:
    System advanced one timestep into the future (np.array)

    Components used: neuromancer.psl.nonautonomous

    Side effects: Keep track of current state of the system (change object attribute).