# til-final

SDK, simulator and documentation for TIL 2022 Robotics
Challenge.

* ``src/``: SDK and simulator packages.
* ``config/``: SDK and simulator sample configuration files.
* ``data/``: Simulator sample data.
* ``docs/``: Documentation source.
* ``stubs/``: Code stubs for participants.


## Install

```sh
pip install .
```

Robomaster SDK will need to be installed separately.

## Build the documentation

Most of the information is available in the Sphinx docs.

```sh
# install dependencies
pip install sphinx sphinx-autoapi sphinx-rtd-theme

# build docs
sphinx-build -b html docs/source docs/build 
```

Access the docs at `docs/build/index.html`.