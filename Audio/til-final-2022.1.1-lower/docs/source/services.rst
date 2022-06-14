Provided Services
-----------------

Several services are provided to enable you to interact with the robot and arena.

This page describes the services in detail. If you use the provided Challenge SDK, you
may skip the service detail sections.


Robotmaster SDK
~~~~~~~~~~~~~~~

The DJI Robomaster SDK allows control of the robot. See `DJI Robomaster SDK documentation
<https://robomaster-dev.readthedocs.io/en/latest/>`_ for details.

.. warning:: 
    The robot pose information provided by the SDK is a result of dead-reckoning,
    and is not aligned to the arena frame. For this reason it cannot be used
    with the rest of the challenge provided services.

    Use the location and pose information provided by the :ref:`localization-service`
    instead.

.. _localization-service:

Localization Service
~~~~~~~~~~~~~~~~~~~~

The localisation service provides the robot's real-time pose and any clues received
at the robot's present location. Clues each comprise an audio file and an audio source
location. The audio may indicate various emotions, and audio that indicate distress
(classified "angry" or "sad") should be prioritised for investigation.

.. tip::
    To use the SDK to obtain robot pose and clues see :py:meth:`tilsdk.localization.LocalizationService.get_pose`

    Example:

    .. code-block:: python

        from tilsdk.localization import *

        ...

        loc_service = LocalizationService()

        ...

        pose, clues = loc_service.get_pose()

.. _localization-service-details:

Service Details
###############

The localisation service shall be available as a HTTP endpoint at 
``http://<host>:<port>/pose``. A ``GET`` request on the localisation service endpoint
shall return a JSON object (Content-Type ``application/json``) of the following schema:

.. code-block:: json

    {
        "pose": {
            "x": 1.0,
            "y": 2.0,
            "z": 3.0
        },
        "clues": [
            {
                "id": 1,
                "location": {
                    "x": 4.0,
                    "y": 5.0
                },
                "audio": "Q29uZ3JhdHVsYXRpb25zLCB5b3UgZm91bmQgYW4gRWFzdGVyIEVnZyE="
            }
        ]
    }


======================= ========= ====== ========================================================================
Field name              Type      UOM    Remarks                                                                 
======================= ========= ====== ========================================================================
``pose.x``              float     m      x-coordinate of robot location relative to arena.                       
``pose.y``              float     m      y-coordinate of robot location relative to arena.                       
``pose.z``              float     deg    Robot heading relative to arena, measured clockwise from x-direction.   
``clues[n].clue_id``    int       N/A    Unique clue ID.                                                         
``clues[n].location.x`` float     m      x-coordinate of location of audio source, relative to arena.            
``clues[n].location.y`` float     m      y-coordinate of location of audio source, relative to arena.            
``clues[n].audio``      string    N/A      Base64 encoded WAV file to be classified.                               
======================= ========= ====== ========================================================================

.. _map-service:

Map Service
~~~~~~~~~~~

The map service provides a static map of the arena as an image. The image represents
an occupancy grid, where non-occupied (i.e. passable) cells are black and occupied
(i.e. non-passable) cells are white. This map does not change, so it only needs to be
retrieved once.

.. tip:: 
    To use the SDK to get the map, see :py:meth:`tilsdk.localization.LocalizationService.get_map`.

    Example:

    .. code-block:: python

        from tilsdk.localization import *

        ...

        loc_service = LocalizationService()

        ...

        map_ = loc_service.map()

.. _map-service-details:

Service Details
###############

The map service shall be available as a HTTP endpoint at ``http://<host>:<port>/map``.
A ``GET`` request on the map service endpoint shall return a JSON object 
(Content-Type ``application/json``) of the following schema:

.. code-block:: json

    {
        "map": {
            "scale": 0.01,
            "grid": "Q29uZ3JhdHVsYXRpb25zLCB5b3UgZm91bmQgYW4gRWFzdGVyIEVnZyE="
        }
    }


============= ========= ========= ================================================================================================================================================================================== 
Field name    Type      UOM       Remarks                                                                                                                                                                           
============= ========= ========= ================================================================================================================================================================================== 
``scale``     float     m/px      Scale of the grid image. A scale of n means each pixel in the grid image represents nxn meters on the physical arena.                                                             
``grid``      string    N/A       Base64 encoded grid image, the value of each pixel indicates the occupancy of that cell. The image is black (value=0) for empty cells and white (value=255) for occupied cells.   
============= ========= ========= ================================================================================================================================================================================== 

.. _reporting-service:

Reporting Service
~~~~~~~~~~~~~~~~~

The reporting service allows the robot to report targets to human rescuers. For the 
purposes of the competition, the submissions to the report service are used for scoring.

.. tip:: 
    To use the SDK to report targets see :py:meth:`tilsdk.reporting.ReportingService.report`.

    Example:

    .. code-block:: python

        from tilsdk.reporting import ReportingService

        ...

        rep_service = ReportingService()

        ...

        rep_service.report(pose, img, targets)


.. _reporting-service-details:

Service Details
###############

The reporting service shall be available as a HTTP endpoint at ``http://<host>:<port>/report``.
To submit targets, perform a ``POST`` request on the reporting service endpoint with a JSON
message body (Content-Type ``application/json``) of the following schema:

.. code-block:: json

    {
        "pose": {
            "x": 1.0,
            "y": 2.0,
            "z": 3.0
        },
        "image": "Q29uZ3JhdHVsYXRpb25zLCB5b3UgZm91bmQgYW4gRWFzdGVyIEVnZyE=",
        "targets": [
            {
                "id": 1,
                "cls": "fallen",
                "bbox": {
                    "x": 300.0, 
                    "y": 200.0,
                    "w": 350.0,
                    "h": 100.0
                }
            }
        ]
    }

======================== ========= ========= ============================================================================== 
Field name               Type      UOM       Remarks                                                                       
======================== ========= ========= ============================================================================== 
``pose.x``               float     m         x-coordinate of robot location relative to arena.                             
``pose.y``               float     m         y-coordinate of robot location relative to arena.                             
``pose.z``               float     deg       Robot heading relative to arena, measured clockwise from x-direction.         
``image``                string    N/A       Base64 encoded image, annotated with bounding boxes and id for each target.   
``targets[n].id``        int       N/A       Target ID, must correspond with that in annotated image.                      
``targets[n].cls``       string    N/A       Result of classification. One of “mobile” or “immobile”.                      
``targets[n].bbox.x``    float     pixels    Bounding box center x-coordinate in image.                                    
``targets[n].bbox.y``    float     pixels    Bounding box center y-coordinate in image.                                    
``targets[n].bbox.w``    float     pixels    Bounding box width.                                                           
``targets[n].bbox.h``    float     pixels    Bounding box height.                                                          
======================== ========= ========= ============================================================================== 
