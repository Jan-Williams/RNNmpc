## Example User Stories

1. The user is John, an academic researcher. John wants to use the tool to conduct research in surrogate modeling. John wants a flexible interface to allow for investigations of a wide variety of models and hyperparameters. He has a strong technical background and is familiar with interfacing with others' research code. 

2. The user is Sarah, a beginning grad student. Sarah wants wants to use the tool to learn about surrogate modeling. Sarah wants a simple interface to get started quickly in the learning process. She has some background but not much.

3. Judith is a post doc who did her Ph.D. studying recurrent neural network architectures. She's interested in performing a study that directly compares many different data-driven control techniques. As a result, she wants software that allows her to easily add functionality/new model architectures. Sarah has extensive experience with software.

4. Billy is a control systems engineer who is interested in incorporating data-driven techniques in his work. Billy has extensive experience developing code in MATLAB, but has limited experience in Python. He wants to baseline data-driven control against more classical techniques for his system of interest. As a result, he needs an easy way to instantiate the governing equations of his system of interest.

5. Isabella is a researcher conducting a study on the reproducibility of machine learning research. She's interested in regenerating the results of a manuscript. She needs fine-grained control of network hyperparameters in order to exactly replicate the conditions of the results published in the paper. Isabella has a background in software development and can easily read/understand others' code, provided it follows best practices for code development in Python.

6. Samuel is a fluid dynamicist interested in applying data-driven control to high-dimensional fluids systems. Simulating the systems he's interested in requires FEM methods that are not are part of this work. As a result, he needs model architectures that can easily accept inputs from an external simulation/data collection environment. Samuel has experience working with Python, but only has limited experience with PyTorch.

7. Lauren is an undergrad studying applied math. Her advisor has instructed her to familiarize herself with PyTorch models and optimizers by surveying existing repositories. She has some experience with Python, and is looking to use this work as a jumping off point for understanding both coding practices and the use of Pytorch. 

8. Dale is an astrodynamicist that studies stationkeeping in the circular restricted three body problem (CR3BP). Long integrations of this system are computationally expensive, and he's interested in using machine learning to cheapen the cost of these integrations. Because of the safety critical nature of his work, he requires more interpretable stationkeeping methods and is only interested in using this work to provide cheap, first tests of new stationkeeping strategies. He mostly works in C and MATLAB, but has some experience with Python. 

## Implied Use Case:
1. Functionality to flexibly generate data for a system of interest. The user provides details about the ODEs or PDEs that govern their system of interest, and the code provides an object that simulates the system while providing a consistent interface for subsequent model training and control evaluation.

2. The implied user is a future developer who wants to perform research of their own introducing new network architectures or methods for minimizing the MPC objective function. As a result, the code needs to have good developer documentation in addition to user docs.
