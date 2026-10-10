# Battery Rangers

![Battery rangers](static/battery_rangers.jpeg)

## Setup

### 0. Clone the repository
Clone the repository with `git clone --recurse-submodules` in order to also clone the submodules. Use `
git submodule update --init --recursive` if you did not clone with submodule. Finally, use `git submodule update` to update the submodules.

#### [REINVENT4](https://github.com/MolecularAI/REINVENT4)
Requires additional setup, follow the instructions in their README. Everything should be installed as a separate venv (i.e. `.reinvnet_venv`) with `uv`. In general, all command from the README should simply be preceeded by `uv`, i.e. `uv pip install ...`.

#### [Mol-AIR](https://github.com/wtaisner/Mol-AIR)
Setup is already done with `uv`.

### 1. Python environment

#### Mamba/Conda
Using either `conda` or `mamba`, run `mamba env create -f environment.yaml` to create an environment.
Use `mamba activate battery` to activate it in the terminal.

#### UV
Assuming you have `uv` installed, you can run `uv sync --all-extras --build` to install the environment.

### 2. [pre-commit](https://pre-commit.com)
Install pre-commit hooks: `pre-commit install`. From now on, it will run check automatically on `git commit`.

You can also run it manually with `pre-commit run --all-files`.

## Directory structure
```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   ├── symmetries     <- The symmetry translation data.
│   ├── sampling       <- Results of the sampling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── environment.yaml   <- The requirements file for reproducing the analysis environment.
│
├── pre-commit.yaml   <- Pre-commit hooks to run on commit. See https://pre-commit.com for details.
│
├── .pylintrc   <- Custom configuration for pylint.
│
├── apps               <- Source code for different modules.
│   ├── core           <- Utility functions and classes.
│   ├── ...            <- Specific module.
```

Roughly following [cookiecuter](https://cookiecutter-data-science.drivendata.org/) template.
