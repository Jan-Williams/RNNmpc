# closed-loop-sim function

## Overview
The function 'closed loop sim', supports various simulation environments and forecasters to help achieve optimal control.

## Features
- Support for multiple simulation environments:
  - Lorenz Control
  - Two Tank Control
  - Stirred Tank Control
  - Spring Mass Control
  - Cylinder Control
- Forecasters:
  - LSTMForecaster
  - Reforecast
  - Linear Forecaster
  - Forecaster
  - Unforested
- Modular and scalable design for custom simulation configurations.

## Prerequisites
1. **Python Version**: Ensure Python 3.8+ is installed.
2. **Dependencies**: Install required Python packages.

## Example Usage With LSTMForecaster

from utils import closed_loop_sim

environment = "LorenzControl"
reference_trajectory = np.zeros((3, 1700))  # Example trajectory
train_data = {
    "U_train": ...,  # Replace with your U_train data
    "S_train": ...,  # Replace with your S_train data
    "O_train": ...,  # Replace with your O_train data
}
hyperparams = {
    "Nr": 1000,
    "dropout_p": 0.2,
    "lags": 10,
    "adam_lr": 0.001
}
control_params = {
    "dev": 100,
    "u_1": 1,
    "u_2": 20,
    "control_horizon": 20,
    "forecast_horizon": 50
}

results = closed_loop_sim(environment, reference_trajectory, "LSTMForecaster", train_data, hyperparams, control_params)

print("Controlled States:", results["controlled_states"])
print("Applied Controls:", results["applied_controls"])