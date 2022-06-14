SDK Installation and Use
========================

Code Organization
-----------------

The ``til-final`` repository is organized as follows:

* ``src/``: SDK and simulator packages. You should not modify this.
* ``config/``: SDK and simulator sample configuration files. You should copy these for your own use.
* ``data/``: Simulator sample data. You should copy these for your own use.
* ``docs/``: Documentation source. You should not modify this.
* ``stubs/``: Code stubs. You should copy these for your own use.

You should create a workspace directory (e.g. ``your_workspace``) and copy the relevant
directories to it.


Installing the SDK and Simulator
--------------------------------

.. warning::
    All the code used for this challenge is written in Python3.
    Python2 is unsupporter. Before you begin, ensure that you 
    are using Python3, and use ``python3`` or ``pip3`` instead
    of ``python`` or ``pip`` as required.


.. todo::
    Update links for repository.

Clone the ``til-final`` repository to your local machine.

.. code-block:: sh

    cd your_workspace
    git clone https://github.com/jehontan/til-final.git


Install the SDK and simulator.

.. code-block:: sh

    pip install ./til-final


This will install the ``tilsdk`` library and ``til-simulator``
executable on your system.

You can verify that the install is successful by executing:

.. code-block:: sh

    pip list | grep til-final


Building the Documentation
--------------------------

To build the documentation requires Sphinx and AutoAPI.

.. code-block:: sh

    pip install sphinx sphinx-autoapi sphinx-rtd-theme


Build the documentation as HTML packages

.. code-block:: sh

    cd your_workspace/til-final
    sphinx-build -b html docs/source docs/build/

To view the documentation open ``docs/build/index.html``. 


Using the SDK
--------------

Once installed, the SDK can be used by simply importing ``tilsdk``.
For detials on the SDK, check out :doc:`services` and :doc:`autoapi/index`.


Using the Simulator
-------------------

To use the simulator run ``til-simulator`` in a terminal. For details on
the simulator see :doc:`simulation`.