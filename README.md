# PTPI-DL-ROM

This repository contains the source code implementation (with examples) of the paper: 

*S. Brivio, S. Fresca, A. Manzoni, [PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order models for nonlinear parametrized PDEs](https://www.sciencedirect.com/science/article/pii/S0045782524006595) (2024).*

### Purpose
Other than including an implementation of the PTPI-DL-ROM paradigm, the purpose of the present library is to provide the users with a low-level, 
highly-configurable and versatile framework for operator learning,
with a particular focus on physics-informed techniques.

### Main dependencies
The library is based on the JAX backend of Keras 3.0. We recommend to install it in a new conda environment:
```
conda create -y -n ptpi python=3.10
conda activate ptpi
pip install -r requirements.txt --no-cache-dir
```
As an alternative procedure, we refer to the instructions of [the official Keras page](https://keras.io/getting_started/) to install Keras 3.0 and its dependencies.

### Instructions
To run the code examples, go to ```./ptpi/examples```, and then run the code relative to the example *example_name* with
```python example_name.py train``` to run both training and testing stages or ```python example_name.py test``` to run only the testing phase.
(note: to run ```./ptpi/examples/test_ad.py``` you do not need any additional input argument).

After running the entire code, the results, along with suitable images and movies, are stored within the newly created folder ```./results/example_name/```.



### Cite
If the present repository and/or the original paper was useful in your research, 
please consider citing

```
@article{brivio2024ptpidlroms,
title = {PTPI-DL-ROMs: Pre-trained physics-informed deep learning-based reduced order models for nonlinear parametrized PDEs},
author = {Simone Brivio and Stefania Fresca and Andrea Manzoni},
journal = {Computer Methods in Applied Mechanics and Engineering},
volume = {432},
pages = {117404},
year = {2024},
doi = {https://doi.org/10.1016/j.cma.2024.117404}
}
```

### Data availability
The datasets used in the paper can be downloaded from [this link](https://drive.google.com/drive/folders/1IwPe_5I6W6a41XYrMVbDgzpspcl0KmXq?usp=sharing).
