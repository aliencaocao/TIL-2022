Introduction
============

In a disaster site SAR there are many potential hazards to human rescuers
(e.g. unstable structures, dangerous chemicals and gases, etc.). In order
to minimise the exposure of human rescuers to hazards, robots may be used
to locate survivors. Mobile survivors may be directed to escape the disaster
site themselves, while immobile survivors may have their location reported
to human rescuers, allowing rescuers to prioritise their rescue and better
plan rescue approaches. While teleoperated robots may serve this purpose,
autonomous robots will allow searching of a larger site in less time with
less personnel.

In this final challenge of Brainhack, you will be introduced to
robotics by developing an autonomous search-and-rescue (SAR) robot.

Your challenge is to develop the autonomy software to direct a robot to
navigate about an arena simulating a building disaster site.

You will make use of the CV and NLP models that they have developed in
previous challenges to detect and classify human targets, submitting high
priority targets to remote “rescuers”.

Robot description
~~~~~~~~~~~~~~~~~

The robot you will use in this challenge is the `DJI Robomaster EP 
<https://www.dji.com/sg/robomaster-ep>`_. For this reason this documentation
should be read in conjunction with the `DJI Robomaster SDK documentation
<https://robomaster-dev.readthedocs.io/en/latest/>`_.

.. figure:: _static/img/robomaster.jpg
    :align: center
    :width: 300px 
    
    DJI Robomaster EP Infantry Configuration

Arena description
~~~~~~~~~~~~~~~~~

.. todo::
    Add arena description.