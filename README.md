# PTPI-DL-ROM

This repository contains the source code implementation (with examples) of the paper: 

*S. Brivio, S. Fresca, A. Manzoni, [PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order models for nonlinear parametrized PDEs](https://arxiv.org/abs/2405.08558) (2024).*

### Purpose
Other than providing an implementation of the PTPI-DL-ROM paradigm, the purpose of the present library is to provide the users with a low-level, 
highly-configurable and versatile framework for operator learning,
with a particular focus on physics-informed techniques.

### Main dependencies
The library is based on the JAX backend of Keras 3.0 and requires
* ```keras == 3.0.5```
* ```jax == 0.4.25```
* ```numpy == 1.26.4```
* ```scipy == 1.12.0```
* ```scikit-learn == 1.3.0```
* ```tqdm == 4.66.4```
* ```matplotlib == 3.8.0```
* ```pillow == 10.2.0```
  
We recommend to follow throughly the instructions of [the official Keras page](https://keras.io/getting_started/) to install Keras 3.0 and its dependencies.

### Instructions
To run the code examples, go to ```./ptpi/examples```, and then run the code relative to the example *example_name* with
```python example_name.py train``` to run both training and testing stages or ```python example_name.py test``` to run only the testing phase.
(note: to run ```./ptpi/examples/test_ad.py``` you do not need any additional input argument).

After running the entire code, the results, along with suitable images and movies, are stored within the newly created folder ```./results/example_name/```.



### Cite
If the present repository and/or the original paper was useful in your research, 
please consider citing

```
@misc{brivio2024ptpidlroms,
      title={PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order models for nonlinear parametrized PDEs}, 
      author={Simone Brivio and Stefania Fresca and Andrea Manzoni},
      year={2024},
      eprint={2405.08558},
      archivePrefix={arXiv},
      primaryClass={math.NA}
}
```

### Data availability
Further info available soon!
