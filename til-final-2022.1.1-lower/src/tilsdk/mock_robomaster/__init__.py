'''
This module provides mock classes for the DJI Robomaster SDK, emulating the interfaces of the SDK.

The mock classes allow interaction with the simulator instead of the physical robot
and can be used for testing.

.. warning::
    This module should not be used during the actual challenge and the actual DJI  SDK should be used.

    .. code-block:: python

        # Use this for testing
        from mocks.robomaster.robot import Robot

        # Use this for actual challenge
        from robomaster.robot import Robot
'''