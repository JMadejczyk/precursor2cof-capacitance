# precursor2cof-capacitance

## Setup

### 1. Python environment

#### Mamba/Conda
Using either `conda` or `mamba`, run `mamba env create -f environment.yaml` to create an environment.
Use `mamba activate battery` to activate it in the terminal.

#### UV
Assuming you have `uv` installed, you can run `uv sync --all-extras --build` to install the environment.

### 2. [pre-commit](https://pre-commit.com)
Install pre-commit hooks: `pre-commit install`. From now on, it will run check automatically on `git commit`.

You can also run it manually with `pre-commit run --all-files`.


