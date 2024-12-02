## Example User Stories

1. The user is John, an academic researcher. John wants to use the tool to conduct research in surrogate modeling. John wants a flexible interface to allow for investigations of a wide variety of models and hyperparameters. He has a strong technical background and is familiar with interfacing with others' research code. 

2. The user is Sarah, a beginning grad student. Sarah wants wants to use the tool to learn about surrogate modeling. Sarah wants a simple interface to get started quickly in the learning process. She has some background but not much.

3. Judith is a post doc who did her Ph.D. studying recurrent neural network architectures. She's interested in performing a study that directly compares many different data-driven control techniques. As a result, she wants software that allows her to easily add functionality/new model architectures. Sarah has extensive experience with software.

4. Billy is a control systems engineer who is interested in incorporating data-driven techniques in his work. Billy has extensive experience developing code in MATLAB, but has limited experience in Python. He wants to baseline data-driven control against more classical techniques for his system of interest. As a result, he needs an easy way to instantiate the governing equations of his system of interest.

5. Isabella is a researcher conducting a study on the reproducibility of machine learning research. She's interested in regenerating the results of a manuscript. She needs fine-grained control of network hyperparameters in order to exactly replicate the conditions of the results published in the paper. Isabella has a background in software development and can easily read/understand others' code, provided it follows best practices for code development in Python.

6. Samuel is a fluid dynamicist interested in applying data-driven control to high-dimensional fluids systems. Simulating the systems he's interested in requires FEM methods that are not are part of this work. As a result, he needs model architectures that can easily accept inputs from an external simulation/data collection environment. Samuel has experience working with Python, but only has limited experience with PyTorch.

7. Lauren is an undergrad studying applied math. Her advisor has instructed her to familiarize herself with PyTorch models and optimizers by surveying existing repositories. She has some experience with Python, and is looking to use this work as a jumping off point for understanding both coding practices and the use of Pytorch. 

8. Dale is an astrodynamicist that studies stationkeeping in the circular restricted three body problem (CR3BP). Long integrations of this system are computationally expensive, and he's interested in using machine learning to cheapen the cost of these integrations. Because of the safety critical nature of his work, he requires more interpretable stationkeeping methods and is only interested in using this work to provide cheap, first tests of new stationkeeping strategies. He mostly works in C and MATLAB, but has some experience with Python. 

9. Alexis is a robotics engineer.
Alexis wants to implement Model Predictive Control (MPC) on a robotic arm to perform precise object manipulation tasks. She requires a package that seamlessly integrates RNN-based predictors with hardware controllers. Alexis has moderate experience in Python and is familiar with control theory concepts.

10. Chris is a systems biology researcher.
Chris studies cell signaling networks and wants to use RNN-MPC for predicting and controlling the behavior of biochemical pathways. Chris needs an intuitive interface that supports custom dynamics functions for ODEs, as the biological systems exhibit nonlinear dynamics. Chris has extensive experience in MATLAB but is new to Python.

11. Dana is a software developer transitioning into data-driven control.
Dana is interested in adding support for Reinforcement Learning (RL) techniques to the MPC framework using RNN predictors. Dana requires developer-friendly documentation and modular code to build upon the existing architecture. She has a strong software development background but limited experience with control theory.

12. Eva is an undergraduate student in mechanical engineering.
Eva wants to use the package for her coursework to model and control the dynamics of a pendulum system. She needs a simple, beginner-friendly interface to quickly set up simulations and test various control strategies. Eva has basic programming skills and is comfortable with Python.

13. Raj is a climate scientist.
Raj works on weather modeling and is exploring MPC with RNNs to optimize renewable energy resource allocation. Raj requires the package to support high-dimensional systems and integrate seamlessly with external data sources such as satellite imagery. He has moderate experience with Python and data analysis tools.

14. Michael is an aerospace engineer.
Michael designs guidance systems for unmanned aerial vehicles (UAVs). He needs the package to simulate complex flight dynamics and use RNN-MPC to optimize fuel consumption during autonomous navigation. The solution must support constraints on control inputs to model real-world actuator limitations.

15. Sophia is a high-performance computing researcher.
Sophia wants to deploy the RNN-MPC package on a distributed computing cluster to solve large-scale optimization problems for power grid management. The package must be parallelizable and optimized for GPUs to handle massive datasets efficiently.

16. Ivan is a financial analyst.
Ivan applies MPC with RNN predictors to model and control portfolio allocations in volatile markets. He requires support for multi-objective optimization, where both risk and return are balanced dynamically based on market forecasts.

17. Ling is an environmental scientist.
Ling works on controlling irrigation systems to optimize water usage in agriculture. She requires an MPC package that integrates seamlessly with IoT devices to collect sensor data and issue real-time control actions.

18. Ahmed is a Ph.D. student studying traffic flow dynamics.
Ahmed uses the package to implement RNN-MPC for optimizing urban traffic signal timings. The tool must support multi-agent systems and allow Ahmed to simulate interactions between different traffic control units.

19. Nina is a neuroscientist.
Nina studies brain network dynamics and needs to control simulated neural circuits using RNN-based predictive models. The package must support highly nonlinear dynamics and provide visualizations of network states and control actions.

20. Theo is a medical device engineer.
Theo uses RNN-MPC to control an insulin pump system for diabetic patients. The package must include safety constraints to ensure the system remains within safe operating conditions at all times.

21. Emma is a quantum physicist.
Emma uses the package to control quantum systems for state preparation tasks. She requires the tool to support high-dimensional state spaces and interact with quantum simulation libraries.

22. Carlos is an industrial process engineer.
Carlos applies the package to control temperature and pressure in chemical reactors. The system must allow custom constraints and objectives to reflect strict safety regulations.

23. Felicia is an energy economist.
Felicia explores using RNN-MPC to optimize electricity trading between suppliers and consumers in a smart grid. She needs the package to integrate with real-time pricing APIs and perform predictive scheduling.

24. Ravi is a gaming AI developer.
Ravi uses the package to implement adaptive NPC behaviors in games, where RNN-MPC predicts player actions and adjusts strategies dynamically.

25. Monica is an educator.
Monica teaches advanced control systems and uses the package to demonstrate the integration of machine learning models with classical control techniques. The tool must have user-friendly visualization capabilities for educational purposes.

## Implied Use Case:
1. Functionality to flexibly generate data for a system of interest. The user provides details about the ODEs or PDEs that govern their system of interest, and the code provides an object that simulates the system while providing a consistent interface for subsequent model training and control evaluation.

2. The implied user is a future developer who wants to perform research of their own introducing new network architectures or methods for minimizing the MPC objective function. As a result, the code needs to have good developer documentation in addition to user docs.
