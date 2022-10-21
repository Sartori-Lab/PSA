# proFSA
`proFSA` is a Python package for performing Finite Strain Analysis (FSA) on atomic structures of protein. The package loads a pair of protein structures, taken as `pdb/cif` files, and applies the FSA formalism adapted to proteins. The results can be saved in `pdb/cif` files. In addition, it also allows to generate some basic plots of FSA related quantities.

## Installation

The easiest way to start using `proFSA` is to insatll it install it through `pip`:

```bash
pip install proFSA
```
<br/>

Alternatively, you can clone this repository. The requirements will then be
- `numpy` (tested with vX.Y)
- `scipy` (tested with vX.Y)
- `biopython` (tested with vX.Y)
- `urllib` (tested with vX.Y)


## Package description

The main folder of the package is `fsa`, which contains the following files

| File                          | Description |
|-------------------------------|-------------|
| ```amp.py```       | Generic implementation of approximate message passing (AMP) <br/> for low-rank matrix factorisation |
| ```amp_hf*.py```   | Launches AMP for various hopfield models |
| ```cpriors.pyx```  | Cython implementation of various computationally intensive prior functions |
| ```hopfield.py```  | Helper functions to create and analyse Hopfield networks |
| ```priors.py```    | Implementation of the various papers analysed in the paper |
| ```se_*.py```      | State evolution for the AMP algorithms |
| ```tests```        | Contains unit tests for the various modules |

In addition, the folder `paper` contains the notebooks necessary to reproduce the results of the article [XXX], a 


## Usage
Go through the basic parts of a simple script.


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

- follow a simpler directory (fsa, examples)
- fix business with tmp/data and all that
- create setup file and all that
- put on pypi
