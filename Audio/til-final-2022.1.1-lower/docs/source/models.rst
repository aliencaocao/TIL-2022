Adding ML models
================

Machine learning models are often developed and trained on
powerful computers with large amount of resources in order
to reduce development time and allow rapid iteration. Frameworks
used for model development and training are also optimized for
ease of use and modularity, and may not be super efficient.

Robots have a limited weight and power and so typically
have much less capable computers. In order to deploy machine 
learning models for robotics applications, it is typically necessary
to perform some optimization on the models. Techniques include 
targeting inference frameworks that are optimized for the target computer
platform, and model distillattion to reduce the deployed model size.

For this challenge you will be required to run both your NLP and CV model
simulataneously on the same computer. While the provided computer is
fairly capable, it is likely that the unoptimized models will exceed
the computer's resources. To enable both models to be deployed together,
we will optimize the models using ONNX.

.. note::
    Optimization with ONNX is entirely optional. You may use your
    models as-is, or perform your own optimizations. If you choose to
    do so, ensure that you implement ``cv_service.py`` and ``nlp_service.py``
    appropriately.

Optimization with ONNX
----------------------

`ONNX <https://onnxruntime.ai/>`_ is a runtime framework that makes it
easy to optimize and accelerate machine learning models.

In order to use the ONNX runtime, export your models in the ONNX format.
For specific instructions on how to do this for your framework, consult 
`the ONNX documentation <https://onnxruntime.ai/docs/get-started/with-python.html>`_.

You can verify that the ONNX model is correct by inspecting it with the
`Netron Visualizer <https://netron.app/>`_. Make sure to open the "Model Properties"
panel, and take note of the input and output names and types.

Adding models to stubs
----------------------

Once exported, copy the ONNX models to the ``stubs/model`` folder. You will
next need to modify ``cv_service.py`` to utilize your models, specifically
``targets_from_image()``. The modifications will depend on your models, 
however in general you will need to perform pre- and post-process to match
your model input and output. You will also need to run the ONNX inference session
using the correct input and output names, as reported in the Netron visualizer.