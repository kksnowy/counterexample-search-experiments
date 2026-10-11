# Vehicle gradient-based counter-example search experiments

## Setup

The experiment code and materials for each specification are in separate directories. Create a virtual environment using Python 3.10 and install the dependencies listed in `requirements.txt`.

## Gradient-based counter-example search

The code for producing counter-examples for each specification using Vehicle is available as a Pytest test file.
e.g. `test_acasXu_search.py`. Each test file creates or loads the relevant neural network, calls the newly implemented
search function, and times how long it takes to find a counter-example. Each also checks that generated counter-examples 
violate the corresponding property, then writes the result to a time-stamped .csv file stored in the specification's
`results` subdirectory.

These test files were run within the Vehicle codebase to obtain the results in the thesis. However, since the counter-example 
search feature is not yet available in the latest version (v0.28.0) of the published Vehicle Python package (pending the next 
version release), these test files cannot be executed here to reproduce the results with this version of Vehicle listed in 
`requirements.txt`. However, the test files are still made available here for the reader to inspect.

Once the next version of the Vehicle package has been released, upgrade to the newer version using

`pip install vehicle-lang --upgrade`

then each test file can be run using Pytest e.g.

`python -m pytest test_acasXu_search.py::<name of relevant test>`

## Finding counter-examples with Marabou

The code used to measure the times taken by Marabou to find counter-examples for MNIST and ACAS Xu properties is available 
in the scripts suffixed with "_verify" e.g. `acasXu_verify.py`. These can be executed here to reproduce the times e.g. via
the command `python acasXu_verify.py`. 

To measure the time for a specific value of epsilon for MNIST, or a specific property for ACAS Xu, the variables 
`EPSILON` and `PROPERTY` need to be modified in the corresponding file. The result is written to a time-stamped .txt file
stored in the specification's `results` subdirectory.

## Neural networks used

Neural networks used for the synthetic specifications are created within its Pytest test file.

Pre-trained neural networks used for the MNIST and ACAS Xu specifications are available in the `models` directory which has 
`onnx` and `pytorch` subdirectories containing the following files:

- `onnx/`
  - `acasXu_1_7.onnx` - one of the 45 neural networks which make up the ACAS Xu system. This file was obtained from the
  		`examples/acasXu` directory within Vehicle's codebase, but was originally from [here](https://github.com/NeuralNetworkVerification/Marabou/tree/master/resources/onnx/acasxu).
  - `mnist-classifier.onnx` - an MNIST classifier network obtained from the `examples/mnist-robustness` directory within Vehicle's 
  		codebase, but was originally from [here](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist).
- `pytorch/`
  - `acasXu_1_7.pth` - the network in `acasXu_1_7.onnx` converted into Pytorch using the onnx2pytorch package.
  - `mnist-classifier.pth` - the network in `mnist-classifier.onnx` also converted into Pytorch.

The onnx networks were used to find counter-examples using Marabou, while their Pytorch versions were used for gradient-based counter-example search in Vehicle.

## Other resources used

The images and labels used for the MNIST experiments are available as `t2-images.idx` and `t2-labels.idx` respectively 
in the `mnist` directory. The first image and label pair are for the digit "7", while the second pair are for the digit "2".
These files were obtained from the `examples/mnist-robustness` directory within the Vehicle codebase, but their original source is
[here](http://yann.lecun.com/exdb/mnist).