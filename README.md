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


## Modules

<dl>
  <dt><code>db</code></dt>
  <dd>Library which lets you read and write to the MySQL database. See the embedded documentation for details.</dd>
  <dt><code>fulltext_processor</code><dt>
  <dd>Runnable module which lets you populate the MySQL database with metadata from the fulltext dataset.</dd>
  <dt><code>preprocessor</code></dt>
  <dd>Runnable module which creates files out of the dataset, fit for consumption by the algorithm script.</dd>
  <dt><code>hybrid</code></dt>
  <dd>Library with a hybrid recommender algorithm.</dd>
  <dt><code>algo1</code></dt>
  <dd>Runnable module which sets up a hybrid recommender system and runs it using the available dataset, by first training and then testing the algorithm. A number of metrics are collected about the algorithm's performance.</dd>
</dl>