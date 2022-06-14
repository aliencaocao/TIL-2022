Simulation & Testing
====================

Simulation is an integral part of the robotics development workflow.
Robot software may be developed in parallel with hardware to speed 
up development. There may be limited copies of the physical robot to
develop and test with. Physical conditions may be difficult and expensive
to create and make reproducible. Simulation overcomes these challenges
by reducing the need to perform physical tests, replacing them with tests
in a configurable virtual environment.

In this challenge a basic simulator (``til-simulator``) is available to you
to assist you in testing your autonomy code. You are encouraged to design
various test environments and test extensively in simulation as access to
the physical challenge arena and robots prior to the actual challenge day
will be limited.

Simulation Setup
~~~~~~~~~~~~~~~~

.. _sim-setup:
.. figure:: _static/img/sim_setup.svg
    :align: center
    
    Simulation setup. Compare :ref:`Challenge setup <challenge-setup>`.

The simulation setup is designed to closely replicate the physical challenge
setup to allow you to test your autonomy code without any modification. This
is accomplished by:

1. Replicating the service endpoints using ``til-simulator``.
2. Providing mock to facilitate isolating desired functions.

Simulator
---------

``til-simulator`` sets up endpoints with the same interface as the
localization and reporting servers, allowing you to use the localization
and reporting service from the SDK without modification as you would in the
actual challenge. It accepts configuration options (see `simulator-configuration`)
to allow injecting your own arena maps and clues. It simulates the robot
movement and provides a visualization of the robot behaviour.

The simulator can also be configured to provide a passthrough proxy of an actual
localization server. This enables you to test with the physical robot and localization
while using your own map and injecting your own clues.

.. note::
    ``til-simulator`` sets up the HTTP endpoints. When running the simulator locally
    on your development machine, your default machine configuration may prevent the
    simulator from communicating with your autonomy code.

    If you face trouble with running the simulation, check that

    1. Your loopback adpater is enabled (on Windows)
    2. Your firewall settings are not blocking traffic to the simulator.

    You can also change the host and port used by the simulator by providing it in a
    config file or via command line option (see :ref:`simulator-configuration`).

.. _mocks:

Mocks
-----

A set of mock modules are provided under the ``mocks`` subdirectory. These mocks
proivide the same interface and are drop in replacements for the Robomaster SDK,
CV and NLP services. By using the mock modules and modifying them to inject
certain behaviour, you can isolate portions of your autonomy code for testing,
allowing you to reproduce test conditions and test edge cases.

.. tip::

    To use a mock, simply replace the import statement of the target module with 
    the equivalent mock import.

    .. code:: python
        
        # Comment out the acutal import
        # from cv.cv_service import CVService

        # Replace with mock import
        from mocks.mock_cv import CVService

.. _simulator-configuration:

Simulator Configuration
~~~~~~~~~~~~~~~~~~~~~~~

The simulator accepts a YAML config file and command line options for configuration.
A sample config is provided in ``config/sim_config.yml``. Available command line 
options can be viewed by running ``til-simulator --help``.

You may use different maps with the simulator. Maps should be provided as black
and white images in PNG format. Free grid positions should be marked black and 
obstacle positions should be marked white. Be sure to specify the map scale used.