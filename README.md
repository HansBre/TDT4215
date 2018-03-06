# TDT4215

Coding style: https://google.github.io/styleguide/pyguide.html

## Usage

### Install dependencies

Unix:

First install python packages for scikit-learn and other dependencies.

Then, install all other dependencies with:
```sh
pipenv --site-packages install
```

Windows:

Just try:

```sh
pipenv install
```

### Run

Run while in the root folder:

```sh
pipenv run python -m src.MODULE_NAME ARGUMENTS
```

where `MODULE_NAME` is the name of a Python module to run (like `algo1`) and `ARGUMENTS` are the arguments you'd like to pass to the module.

Use `--help` to see available arguments.