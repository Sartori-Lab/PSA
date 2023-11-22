# PSA
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`PSA` is a Python package for performing Protein Strain Analysis (PSA). The package loads a pair of protein structures, taken as `pdb/cif` files, and applies the Finite Strain Analysis formalism adapted to proteins. The results can be saved in `pdb/cif` files. In addition, it also allows to generate some basic plots of FSA related quantities.

## Installation

The easiest way to start using `PSA` is to install it install it through `pip` as follows

```bash
pip install PSA-current_version.whl
```
You can find the latest version `whl` file in the release section.

The requirements are
- `numpy` (tested with v1.25.2)
- `scipy` (tested with v1.11.2)
- `numba` (tested with v0.58.0)
- `biopython` (tested with v1.81)
- `urllib3` (tested with v1.26.7)
- `matplotlib` (tested with v3.8.0)
  
## Package description

The folder package is `psa`, which contains the following files

| File                          | Description |
|-------------------------------|-------------|
| ```load.py```       | Functions related to loading, parsing and modifying `pdb/cif` structure files |
| ```forms.py```   | Functions that generate elementary deformations on a cylinder shape, aimed at testing the method |
| ```sequence.py```  | All functions related to aligning or comparing protein sequences |
| ```spatial.py```  | Functions to manipulate structures in space |
| ```elastic.py```    | Functions that perform Protein Strain Analysis |

In addition, the folder `examples` contains a notebook showing a minimal working example of how to use `PSA`, and another notebook performing basic tests on elementary deformations of a cylinder.


## Usage
Here is a simple example showing how to calculate strain for a pair of structures:

```python
import psa.load as load
import psa.sequence as seq
import psa.elastic as elastic

# Load structures
rel_pps = load.single_structure(name = '6FKF')
def_pps = load.single_structure(name = '6FKH')

# Pairwise sequence alignment
com_res, rel_dict, def_dict = seq.pairwise_alignment(rel_pps, def_pps)

# Load coordinates of atoms
rel_xyz, rel_labels = load.coordinates(rel_pps,
                                       com_res,
                                       rel_dict)
def_xyz, def_labels = load.coordinates(def_pps,
                                       com_res,
                                       def_dict)

# Calculating the neighbourhood of each atom
weights = elastic.compute_weights_fast([rel_xyz, def_xyz], 
                                       "intersect",
                                       [10.])

# Calculating the deformation gradient
F = elastic.deformation_gradient_fast(weights,
                                      rel_xyz,
                                      def_xyz)

# Obtaining readouts: principal stretches squared
# and principal directions
gam_l, gam_n = elastic.lagrange_strain(F)
stretches, stretch_axis = elastic.principal_stretches_from_g(gam_n)
```


## Citation

If you are using this package as part of an academic project, please cite the following reference

```bash
@article{sartori2023evolutionary,
  title={Evolutionary conservation of mechanical strain distributions in functional transitions of protein structures},
  author={Sartori, Pablo and Leibler, Stanislas},
  journal={bioRxiv},
  pages={2023--02},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

