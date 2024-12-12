# Components:

## 1. Forecast. 

    Inputs:
    current state of the system (torch.DoubleTensor)

    current sensor measurements (torch.DoubleTensor)

    subsequent control inputs (torch.DoubleTensor)

    Outputs: 
    forecast of system dynamics (torch.DoubleTensor)

    Components used:
    module forward function (essentially just wrapped in a for loop)

    No side effects.



## 2. Simulator(ODE)
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

## 3. Simuate closed loop
    Inputs:
    trained forecaster (ESNForecaster, LSTMForecaster, etc.)

    instantiated controller object with control weightings (MPController)

    simulation environment(ODE)

    Outputs:
    simulated closed loop trajectory (np.array)

    Side effects: None.

## 4. (Experimental) S5 model implemented in Jax
    Inputs: 
    dimensionality of S5 layers (int)

    number of stacked S5 layers (int)

    input dimension (int, dim. control input + dim. state measurement)

    output dimension (int, dim. state measurements)

    Optional arguments: initialization method, nonlinearity

    Outputs:
    initialized model in Jax (nnx.Module)

    Side effects:
    update parameters of S5 model

    Components used: nnx.Module

Caveat: Although existing components of this work are all implemented in PyTorch, implementing a modified S5 architecture in Jax has a variety of advantages for downstream applications (namely, the more functional API of Jax will make other forms of control easier to implement). At this time, this experimental model will not function with existing methods/protocols.

## 5. (Experimental) Train S5 model
    Inputs:
    training data (jnp.array)

    optimizer (optax optimizers)

    Outputs: 
    None

    Side effects: 
    update parameters of S5 model

    Components used:
    S5 model    