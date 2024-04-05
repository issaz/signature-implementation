# signature-implementation

Code accompanying the chapter "Practical implementation of the signature method".

This repository consists of three notebooks:

1) ```worked_example_phi_kernels.ipynb```: Notebook accompanying Sections 3.1.3 and 3.3. Covers level contributions and combinations, and works through the example of minimising the Type II error between two scaled Brownian motions.
2) ```n_samples_v_length.ipynb```: Notebook accompanying results from Section 3.2.1. Plots of the expected Type II error for different discretisation frequencies and batch sizes.
3) ```find_optimal_measure.ipynb```: Addendum notebook not discussed in the body of the work. A potential algorithm to find the scaling which minimises the expected Type II error between two sets of sample paths.

All required packages are present in ```requirements.txt```.