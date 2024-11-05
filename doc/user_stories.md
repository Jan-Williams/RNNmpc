## Example User Stories

1. The user is John, an academic researcher. John wants to use the tool to conduct research in surrogate modeling. John wants a flexible interface to allow for investigations of a wide variety of models and hyperparameters. He has a strong technical background and is familiar with interfacing with others' research code. 

2. The user is Sarah, a beginning grad student. Sarah wants wants to use the tool to learn about surrogate modeling. Sarah wants a simple interface to get started quickly in the learning process. She has some background but not much.

3. Judith is a post doc who did her Ph.D. studying recurrent neural network architectures. She's interested in performing a study that directly compares many different data-driven control techniques. As a result, she wants software that allows her to easily add functionality/new model architectures. Sarah has extensive experience with software.


## Implied Use Case:
1. Functionality to flexibly generate data for a system of interest. The user provides details about the ODEs or PDEs that govern their system of interest, and the code provides an object that simulates the system while providing a consistent interface for subsequent model training and control evaluation.

2. The implied user is a future developer who wants to perform research of their own introducing new network architectures or methods for minimizing the MPC objective function. As a result, the code needs to have good developer documentation in addition to user docs.
