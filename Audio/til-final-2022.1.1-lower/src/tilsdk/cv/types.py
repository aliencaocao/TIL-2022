from collections import namedtuple

BoundingBox = namedtuple('BoundingBox', ['x', 'y', 'w', 'h'])
'''Bounding box (bbox).

.. py:attribute:: x
    :type: float

    bbox center x-coordinate.

.. py:attribute:: y
    :type: float

    bbox center y-coordinate.

.. py:attribute:: w
    :type: float

    bbox width.

.. py:attribute:: h
    :type: float

    bbox height.
'''

DetectedObject = namedtuple('DetectedObject', ['id', 'cls', 'bbox'])
'''Detected target object.

.. py:attribute:: id

    Unique target id.

.. py:attribute:: cls

    Target classification.

.. py:attribute:: bbox
    :type: BoundingBox

    Bounding box of target.
'''