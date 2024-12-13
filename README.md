# RNN model predictive control
#### Authors: Jan Williams, Deyang Zheng, and Stewart Lamon

This package includes a variety of recurrent neural network architectures that can be used in data-driven model predictive control. For some minimal runnable examples of the code developed as part of CSE583, please see scripts/ExampleNotebooks/projectdemo.ipynb. We had 3 main objectives for this project, summarized below:

1. (Deyang) Implementation of closed loop simulation.

        Primarily responsible for:
        src/RNNmpc/utils.py
        tests/test_closed_loop_sim.py
        scripts/ExampleNotebooks/projectdemo.ipynb
        README.md
        doc/components.md
        doc/user_stories.md
        doc/CSE583FinalPresentation.pdf

2. (Stewart) Implementation of ODE and Neuromancer integrator classes.

        Primarily responsible for:
        src/RNNmpc/Simulators/neuromancercontrol.py
        src/RNNmpc/Simulators/odecontrol.py
        tests/Simulators/test_odecontrol.py
        tests/Simulators/test_neuromancercontrol.py
        scripts/ExampleNotebooks/projectdemo.ipynb
        README.md
        doc/components.md
        doc/user_stories.md
        doc/CSE583FinalPresentation.pdf


3. (Jan) Implementation of S5forecaster model and associated components. 

        Primarily responsible for:

        src/RNNmpc/Forecasters/s5_forecaster.py
        tests/Forecasters/test_s5_forecaster.py
        environment.yml
        pyproject.toml
        .github/workflows/python-package-conda.yml
        scripts/ExampleNotebooks/projectdemo.ipynb
        README.md
        doc/components.md
        doc/user_stories.md
        doc/CSE583FinalPresentation.pdf

