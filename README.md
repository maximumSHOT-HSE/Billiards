## Description

Takes two dimensional array of floats from an input file which
is single positional arguments of script and prints into stdout.

### Input file format

`N` `M`

`a11`, `a12`, ..., `a1M`

.

.

.

`aN1`, `aN2`, ..., `aNM`

## Run

* `export PYTHONPATH=$PYTHONPATH:$(pwd)`

### Run script
* `python3 main.py`

### Run tests
* `python3 -m unittest test/test_rotate.py`
