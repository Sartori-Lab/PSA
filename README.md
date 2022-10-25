# proFSA
`proFSA` is a Python package for performing Finite Strain Analysis (FSA) on atomic structures of proteins. The package loads a pair of protein structures, taken as `pdb/cif` files, and applies the FSA formalism adapted to proteins. The results can be saved in `pdb/cif` files. In addition, it also allows to generate some basic plots of FSA related quantities.

## Installation

The easiest way to start using `proFSA` is to insatll it install it through `pip` as follows

```bash
pip install proFSA
```
 The requirements are

- `numpy` (tested with vX.Y)
- `scipy` (tested with vX.Y)
- `numba` (tested with vX.Y)
- `biopython` (tested with vX.Y)
- `urllib` (tested with vX.Y)


## Package description

The folder package is `fsa`, which contains the following files

| File                          | Description |
|-------------------------------|-------------|
| ```load.py```       | Functions related to loading, parsing and modifying `pdb/cif` structure files |
| ```forms.py```   | Functions that generate elementary deformations on a cylinder shape, aimed at testing the method |
| ```sequence.py```  | All functions related to aligning or comparing protein sequences |
| ```spatial.py```  | Functions to manipulate structures in space |
| ```elastic.py```    | Functions that perform Finite Strain Analysis |

In addition, the folder `examples` contains a notebook showing a minimal working example of how to use `proFSA`, and another notebook performing basic tests on elementary deformations of a cylinder.


## Usage
A simple example showing how to load a pair of structures:

```python
import fsa.load as load
import fsa.sequence as seq

pps_pair =  load.structure_pair('6FKF', '6FKH')   # load pdb pair
com_res, ind_dict = seq.common_residues() # find common residues, and translation dictionary
xyz_pairs, _ = load.coordinates() # load coordinates
elastic.deformation_gradient
elastic.rotations()
```


## Citation

If you are using this package as part of an academic project, please cite the following reference

```bash
@article{XXXYYYZZZ,
         author = {Sartori, Pablo and Leibler, Stanislas},
         title = {},
         journal = {ArXiv e-prints},
         archivePrefix = \"arXiv\",
         eprint = {2022.XYZ},
         year = 2020}
```

## ToDo

- add cylinder file notebook
- the chunk that follows should be a function
``` python
uni_ids = [None] * len(rel_pps)
my_ref_seqs = seq.generate_reference(rel_pps, uni_ids)
rel_idx = seq.align(rel_pps, my_ref_seqs)
def_idx = seq.align(def_pps, my_ref_seqs)
rel_dict = seq.aligned_dict(rel_pps, rel_idx)
def_dict = seq.aligned_dict(def_pps, def_idx)
com_res = seq.common(rel_dict, def_dict)
``